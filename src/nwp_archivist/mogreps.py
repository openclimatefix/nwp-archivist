"""Reading MOGREPS-UK files from the Met Office's public S3 bucket, by byte range.

Each file holds one variable at one lead time for all three members of a run, as an HDF5 (NetCDF4)
dataset chunked by member and 128 by 128 cells, and compressed with deflate. A file is 4 MB to
106 MB, and the recorder needs one member's crop of it. So the reader parses the file's chunk index
through h5py, using a block-cached file object, and then downloads only the chunks that overlap the
archive's grid, in a few coalesced range requests. The chunk downloads happen outside h5py, which
holds a global lock while it reads.

As with DWD, a file that is absent, truncated, unreadable, or not the file its address promised is
"not yet". Nothing here raises for the provider misbehaving.
"""

import io
import logging
import threading
import zlib
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Final

import h5py
import numpy as np

from nwp_archivist.cache import RunGrid
from nwp_archivist.dwd import Fetcher
from nwp_archivist.products import (
    BOX_LAT_MAX,
    BOX_LAT_MIN,
    BOX_LON_MAX,
    BOX_LON_MIN,
    MOGREPS_BASE_URL,
    ExpectedFile,
    Product,
)
from nwp_archivist.source import Cropped, NotYet

logger = logging.getLogger(__name__)

# HDF5 reads the chunk index in small pieces scattered through the file, so a block is small.
BLOCK_BYTES: Final[int] = 64 * 1024
# Chunks of one row of chunks sit next to each other, with a gap of about 50 KB to the next row.
# Two chunks closer than this are fetched in one request.
COALESCE_GAP_BYTES: Final[int] = 64 * 1024
# netCDF's default fill value is 9.97e36. A cell holding it was never written.
FILL_THRESHOLD: Final[float] = 1e30
STATIC_NAMES: Final[frozenset[str]] = frozenset({"clat", "clon"})


class _BlockFile(io.RawIOBase):
    """A read-only file object over a remote file, fetching and keeping whole blocks."""

    def __init__(self, *, url: str, fetcher: Fetcher) -> None:
        """Wrap `url`. The first read fetches block 0, which also gives the file's size."""
        super().__init__()
        self._url = url
        self._fetcher = fetcher
        self._blocks: dict[int, bytes] = {}
        self._position = 0
        self._size: int | None = None
        # The reason for the first failed request. h5py calls back into `seek` and `readinto` while
        # it handles an exception from either, which corrupts the interpreter's error state, so a
        # failed request looks like the end of the file and the reason is kept here instead.
        self.failure: str | None = None

    def _block(self, index: int) -> bytes:
        if index not in self._blocks:
            fetched = self._fetcher.get_range(
                self._url, start=index * BLOCK_BYTES, stop=(index + 1) * BLOCK_BYTES
            )
            if isinstance(fetched, NotYet):
                self.failure = self.failure or fetched.reason
                self._size = self._size or 0
                return b""
            self._blocks[index], self._size = fetched
        return self._blocks[index]

    def prefetch(self) -> NotYet | None:
        """Fetch block 0, so that a missing file is reported before h5py opens it."""
        self._block(0)
        return None if self.failure is None else NotYet(self.failure)

    def _file_size(self) -> int:
        if self._size is None:
            self._block(0)
        return self._size or 0

    def readable(self) -> bool:
        """Say that the file can be read."""
        return True

    def seekable(self) -> bool:
        """Say that the file can be repositioned."""
        return True

    def tell(self) -> int:
        """The current position."""
        return self._position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        """Move to `offset`, counted from the start, the current position, or the end."""
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self._position, io.SEEK_END: self._file_size()}
        self._position = base[whence] + offset
        return self._position

    def readinto(self, buffer: bytearray | memoryview) -> int:  # ty: ignore[invalid-method-override]
        """Fill `buffer` from the current position, returning how many bytes were read."""
        wanted = min(len(buffer), self._file_size() - self._position)
        if wanted <= 0:
            return 0
        pieces: list[bytes] = []
        position = self._position
        while position < self._position + wanted:
            block = self._block(position // BLOCK_BYTES)
            start = position % BLOCK_BYTES
            take = min(len(block) - start, self._position + wanted - position)
            if take <= 0:
                break
            pieces.append(block[start : start + take])
            position += take
        data = b"".join(pieces)
        buffer[: len(data)] = data
        self._position += len(data)
        return len(data)


@dataclass
class _OpenFile:
    """A file opened by one thread, kept so that the next member of the same file reuses it."""

    url: str
    file: h5py.File
    reader: _BlockFile


def _attr_text(value: object) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


def _model_version(file: h5py.File) -> int:
    """The Unified Model version, such as 13.8 as 1308, or 0 if the file does not say."""
    try:
        major, _, minor = _attr_text(file.attrs["um_version"]).partition(".")
        return int(major) * 100 + int(minor)
    except KeyError, ValueError:
        return 0


@dataclass(frozen=True)
class LambertAzimuthalEqualArea:
    """The ellipsoidal Lambert azimuthal equal-area projection, inverted to longitude and latitude.

    This follows Snyder, "Map Projections: A Working Manual" (1987), equations 3-4 to 3-17 and
    21-15 to 21-28. It is not `pyproj`, because importing `pyproj`
    after `eccodes` corrupts the heap when the process exits.

    Attributes:
        latitude_origin: The latitude of the projection origin, in degrees.
        longitude_origin: The longitude of the projection origin, in degrees.
        semi_major: The ellipsoid's semi-major axis, in metres.
        semi_minor: The ellipsoid's semi-minor axis, in metres.
    """

    latitude_origin: float
    longitude_origin: float
    semi_major: float
    semi_minor: float

    @classmethod
    def from_file(cls, file: h5py.File) -> LambertAzimuthalEqualArea:
        """Read the projection from the file's grid mapping variable."""
        mapping = file["lambert_azimuthal_equal_area"].attrs
        if _attr_text(mapping["grid_mapping_name"]) != "lambert_azimuthal_equal_area":
            message = "the grid is not a Lambert azimuthal equal-area grid"
            raise ValueError(message)
        return cls(
            latitude_origin=float(mapping["latitude_of_projection_origin"][0]),
            longitude_origin=float(mapping["longitude_of_projection_origin"][0]),
            semi_major=float(mapping["semi_major_axis"][0]),
            semi_minor=float(mapping["semi_minor_axis"][0]),
        )

    def inverse(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Convert grid coordinates in metres to longitude and latitude in degrees."""
        a = self.semi_major
        e2 = 1 - (self.semi_minor / a) ** 2
        e = np.sqrt(e2)

        def q(sin_latitude: np.ndarray | float) -> np.ndarray | float:
            return (1 - e2) * (
                sin_latitude / (1 - e2 * sin_latitude**2)
                - np.log((1 - e * sin_latitude) / (1 + e * sin_latitude)) / (2 * e)
            )

        latitude_0 = np.radians(self.latitude_origin)
        q_pole = q(1.0)
        beta_0 = np.arcsin(q(np.sin(latitude_0)) / q_pole)
        radius_q = a * np.sqrt(q_pole / 2)
        d = (
            a
            * np.cos(latitude_0)
            / np.sqrt(1 - e2 * np.sin(latitude_0) ** 2)
            / (radius_q * np.cos(beta_0))
        )
        rho = np.hypot(x / d, d * y)
        safe_rho = np.where(rho == 0, 1.0, rho)
        c = 2 * np.arcsin(np.clip(rho / (2 * radius_q), -1, 1))
        beta = np.where(
            rho == 0,
            beta_0,
            np.arcsin(
                np.clip(
                    np.cos(c) * np.sin(beta_0) + d * y * np.sin(c) * np.cos(beta_0) / safe_rho,
                    -1,
                    1,
                )
            ),
        )
        longitude = np.radians(self.longitude_origin) + np.arctan2(
            x * np.sin(c),
            d * rho * np.cos(beta_0) * np.cos(c) - d**2 * y * np.sin(beta_0) * np.sin(c),
        )
        latitude = (
            beta
            + (e2 / 3 + 31 * e2**2 / 180 + 517 * e2**3 / 5040) * np.sin(2 * beta)
            + (23 * e2**2 / 360 + 251 * e2**3 / 3780) * np.sin(4 * beta)
            + (761 * e2**3 / 45360) * np.sin(6 * beta)
        )
        return np.degrees(longitude), np.degrees(latitude)


def box_rectangle(
    *, x: np.ndarray, y: np.ndarray, projection: LambertAzimuthalEqualArea
) -> RunGrid:
    """Choose the rectangle of grid cells that contains the crop box.

    Args:
        x: The cell-centre x coordinates in metres, one per column.
        y: The cell-centre y coordinates in metres, one per row.
        projection: Converts grid coordinates to longitude and latitude.

    Returns:
        The grid, whose kept cells are the smallest rectangle of rows and columns containing every
        cell whose centre lies inside the box (edges included), flattened row by row.
    """
    columns, rows = np.meshgrid(x.astype(np.float64), y.astype(np.float64))
    lon, lat = projection.inverse(columns, rows)
    inside = (
        (lat >= BOX_LAT_MIN) & (lat <= BOX_LAT_MAX) & (lon >= BOX_LON_MIN) & (lon <= BOX_LON_MAX)
    )
    row_hits, column_hits = np.nonzero(inside)
    first_row, last_row = int(row_hits.min()), int(row_hits.max())
    first_column, last_column = int(column_hits.min()), int(column_hits.max())
    row_range = np.arange(first_row, last_row + 1)
    column_range = np.arange(first_column, last_column + 1)
    cell_index = np.add.outer(row_range * len(x), column_range).ravel().astype(np.int32)
    window = (slice(first_row, last_row + 1), slice(first_column, last_column + 1))
    return RunGrid(
        n_points=len(x) * len(y),
        cell_index=cell_index,
        statics={
            "clat": lat[window].ravel().astype(np.float32),
            "clon": lon[window].ravel().astype(np.float32),
        },
        shape=(len(row_range), len(column_range)),
    )


def mogreps_url(*, base_url: str, init_time: datetime, step_minutes: int, parameter: str) -> str:
    """The address of one file: `.../YYYY/MM/DD/THHMMZ/<valid time>-PT<lead>-<variable>.nc`."""
    valid = init_time + timedelta(minutes=step_minutes)
    lead = f"PT{step_minutes // 60:04d}H{step_minutes % 60:02d}M"
    return (
        f"{base_url}/{init_time:%Y/%m/%d}/T{init_time:%H%M}Z/"
        f"{valid:%Y%m%dT%H%MZ}-{lead}-{parameter}.nc"
    )


class MogrepsSource:
    """The Met Office's MOGREPS-UK bucket: a file per variable and lead time, with all members."""

    def __init__(self, *, fetcher: Fetcher, base_url: str = MOGREPS_BASE_URL) -> None:
        """Build a source.

        Args:
            fetcher: Downloads byte ranges.
            base_url: The bucket's address up to the `YYYY/MM/DD` directories, replaceable so that
                tests can point elsewhere.
        """
        self.fetcher = fetcher
        self.base_url = base_url
        self._local = threading.local()

    def expected_files(self, product: Product, init_time: datetime) -> list[ExpectedFile]:
        """Every file a run should contain, one per member, ordered by variable and lead time.

        The three members of one file are consecutive, so the thread that fetches them reuses the
        open file.
        """
        return [
            ExpectedFile(
                field=field,
                member=member,
                step_minutes=step,
                url=mogreps_url(
                    base_url=self.base_url,
                    init_time=init_time,
                    step_minutes=step,
                    parameter=field.parameter,
                ),
                init_time=init_time,
            )
            for field in product.fields
            for step in field.steps_minutes
            for member in product.members
        ]

    def sequence_key(self, file: ExpectedFile) -> tuple[str, ...]:
        """One variable, all its members, is a sequence."""
        return (file.field.variable,)

    def static_names(self, product: Product) -> set[str]:
        """The grid fields the archive stores once."""
        return set(STATIC_NAMES)

    def _open(self, url: str) -> _OpenFile | NotYet:
        """Open a file, or reuse the one this thread opened last."""
        held: _OpenFile | None = getattr(self._local, "held", None)
        if held is not None and held.url == url:
            return held
        self._forget()
        reader = _BlockFile(url=url, fetcher=self.fetcher)
        missing = reader.prefetch()
        if missing is not None:
            return missing
        try:
            file = h5py.File(reader, "r")
        except (OSError, ValueError, KeyError, RuntimeError) as error:
            return NotYet(reader.failure or f"unreadable: {error}")
        self._local.held = _OpenFile(url=url, file=file, reader=reader)
        return self._local.held

    def _forget(self) -> None:
        """Close the file this thread holds open, if any."""
        held: _OpenFile | None = getattr(self._local, "held", None)
        if held is not None:
            try:
                held.file.close()
            except OSError, ValueError, RuntimeError:
                logger.debug("closing a file failed", exc_info=True)
            self._local.held = None

    def _unreadable(self, error: Exception) -> NotYet:
        """Describe a failure to read the file this thread holds, and drop that file."""
        held: _OpenFile | None = getattr(self._local, "held", None)
        reason = (held.reader.failure if held is not None else None) or f"unreadable: {error}"
        self._forget()
        return NotYet(reason)

    def fetch_grid(
        self,
        *,
        product: Product,
        init_time: datetime,
        only: set[str] | None,
        previous: RunGrid | None,
    ) -> RunGrid | NotYet:
        """Read the grid coordinates from a lead-time-0 file of the run and choose the crop."""
        files = self.expected_files(product, init_time)
        reference = next(file for file in files if file.step_minutes == 0)
        try:
            opened = self._open(reference.url)
            if isinstance(opened, NotYet):
                return opened
            file = opened.file
            grid = box_rectangle(
                x=file["projection_x_coordinate"][:],
                y=file["projection_y_coordinate"][:],
                projection=LambertAzimuthalEqualArea.from_file(file),
            )
            return NotYet(opened.reader.failure) if opened.reader.failure else grid
        except (OSError, KeyError, ValueError, RuntimeError, ArithmeticError) as error:
            return self._unreadable(error)

    def fetch(self, file: ExpectedFile, grid: RunGrid) -> Cropped | NotYet:
        """Download, check, and crop one member of one file."""
        try:
            opened = self._open(file.url)
            if isinstance(opened, NotYet):
                return opened
            result = self._read(opened.file, file, grid)
            return NotYet(opened.reader.failure) if opened.reader.failure else result
        except (OSError, KeyError, ValueError, RuntimeError, zlib.error) as error:
            return self._unreadable(error)

    def _read(self, opened: h5py.File, file: ExpectedFile, grid: RunGrid) -> Cropped | NotYet:
        """Check a file against its address, then download the chunks that overlap the grid."""
        field = file.field
        if field.short_name not in opened:
            return NotYet(f"no {field.short_name} variable")
        dataset = opened[field.short_name]
        n_points = dataset.shape[-1] * dataset.shape[-2]
        if n_points != grid.n_points or grid.shape is None:
            return Cropped(np.empty(0, np.float32), n_points, 0)
        checks = {
            "lead time": int(opened["forecast_period"][()]) == file.step_minutes * 60,
            "run": int(opened["forecast_reference_time"][()]) == int(file.init_time.timestamp()),
            "member": file.member is not None and file.member <= dataset.shape[0],
            "compression": dataset.compression == "gzip" and not dataset.shuffle,
        }
        for name, good in checks.items():
            if not good:
                return NotYet(f"mismatched file: wrong {name}")
        if field.level is None:
            leading: tuple[int, ...] = ()
        else:
            heights = np.asarray(opened["height"][:])
            found = np.flatnonzero(heights == field.level)
            if len(found) != 1:
                return NotYet(f"mismatched file: no {field.level} m level")
            leading = (int(found[0]),)
        member = (file.member or 1) - 1
        values = self._download_crop(dataset, url=file.url, leading=(member, *leading), grid=grid)
        if isinstance(values, NotYet):
            return values
        member_ids = tuple(int(number) for number in opened["realization"][:])
        return Cropped(values, n_points, _model_version(opened), member_ids)

    def _download_crop(
        self, dataset: h5py.Dataset, *, url: str, leading: tuple[int, ...], grid: RunGrid
    ) -> np.ndarray | NotYet:
        """Fetch the chunks under the grid's rectangle and cut the rectangle out of them."""
        rows, columns = grid.shape or (0, 0)
        width = dataset.shape[-1]
        first_row, first_column = divmod(int(grid.cell_index[0]), width)
        chunk_rows, chunk_columns = dataset.chunks[-2:]
        row_chunks = range(first_row // chunk_rows, (first_row + rows - 1) // chunk_rows + 1)
        column_chunks = range(
            first_column // chunk_columns, (first_column + columns - 1) // chunk_columns + 1
        )
        chunk_size = chunk_rows * chunk_columns * dataset.dtype.itemsize
        located: list[tuple[int, int, int, int]] = []  # (offset, size, row chunk, column chunk)
        for row_chunk in row_chunks:
            for column_chunk in column_chunks:
                info = dataset.id.get_chunk_info_by_coord(
                    (*leading, row_chunk * chunk_rows, column_chunk * chunk_columns)
                )
                if info.filter_mask != 0:
                    return NotYet("mismatched file: unexpected chunk filter")
                if info.byte_offset is not None:
                    located.append((info.byte_offset, info.size, row_chunk, column_chunk))
        mosaic = np.full(
            (len(row_chunks) * chunk_rows, len(column_chunks) * chunk_columns),
            np.nan,
            dtype=np.float32,
        )
        for group in _coalesce(sorted(located)):
            start, stop = group[0][0], group[-1][0] + group[-1][1]
            fetched = self.fetcher.get_range(url, start=start, stop=stop)
            if isinstance(fetched, NotYet):
                return fetched
            body = fetched[0]
            if len(body) != stop - start:
                return NotYet("short range")
            for offset, size, row_chunk, column_chunk in group:
                raw = zlib.decompress(body[offset - start : offset - start + size])
                if len(raw) != chunk_size:
                    return NotYet("mismatched file: unexpected chunk size")
                chunk = np.frombuffer(raw, dtype=dataset.dtype).reshape(chunk_rows, chunk_columns)
                top = (row_chunk - row_chunks[0]) * chunk_rows
                left = (column_chunk - column_chunks[0]) * chunk_columns
                mosaic[top : top + chunk_rows, left : left + chunk_columns] = chunk
        top = first_row - row_chunks[0] * chunk_rows
        left = first_column - column_chunks[0] * chunk_columns
        crop = mosaic[top : top + rows, left : left + columns].ravel()
        crop[crop >= FILL_THRESHOLD] = np.nan
        return crop


def _coalesce(
    located: list[tuple[int, int, int, int]],
) -> list[list[tuple[int, int, int, int]]]:
    """Group chunks, sorted by offset, so that each group is one request."""
    groups: list[list[tuple[int, int, int, int]]] = []
    for chunk in located:
        if groups and chunk[0] - (groups[-1][-1][0] + groups[-1][-1][1]) <= COALESCE_GAP_BYTES:
            groups[-1].append(chunk)
        else:
            groups.append([chunk])
    return groups
