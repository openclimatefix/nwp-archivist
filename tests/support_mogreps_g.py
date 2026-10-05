"""Test helpers for MOGREPS-G: synthetic HDF5 files on a small regular lat/lon grid."""

import io
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime

import h5py
import httpx
import numpy as np

from nwp_archivist.dwd import Fetcher
from nwp_archivist.mogreps_g import MogrepsGSource, box_rectangle
from nwp_archivist.products import BOX_LAT_MIN, BOX_LON_MIN, ExpectedFile, Field, Product

BUCKET_URL = "https://bucket.test/global-ensemble"
INIT = datetime(2026, 9, 23, 0, tzinfo=UTC)
REALIZATIONS = tuple(range(4))

# A small grid of 20 rows by 24 columns at 1.5 degree spacing, cut into 8 by 8 cell chunks, and
# placed so that the crop box keeps a rectangle inside it on every side.
N_ROWS = 20
N_COLUMNS = 24
CHUNK = 8
LATITUDE = BOX_LAT_MIN - 6.0 + np.arange(N_ROWS, dtype=np.float32) * 1.5
LONGITUDE = BOX_LON_MIN - 6.0 + np.arange(N_COLUMNS, dtype=np.float32) * 1.5
GRID = box_rectangle(latitude=LATITUDE, longitude=LONGITUDE)
KEPT_ROWS, KEPT_COLUMNS = GRID.shape or (0, 0)
FIRST_ROW, FIRST_COLUMN = divmod(int(GRID.cell_index[0]), N_COLUMNS)
HEIGHT_M = 100

TINY_MOGREPS_G = Product(
    name="tiny-mogreps-g",
    provider="test",
    licence="test",
    cycle_hours=6,
    n_members=len(REALIZATIONS),
    fields=(
        Field("sw", "radiation_flux_in_shortwave_total_downward_at_surface", "sw_data", (0, 60)),
        Field("wind_100m", "wind_speed_on_height_levels", "wind_speed", (0, 60), HEIGHT_M),
    ),
    hhl_levels=(),
    start_delay_hours=6.5,
    deadline_hours=24.0,
    source="mogreps-g",
    live_hours=24.0,
    has_realizations=True,
)


def native_values(file: ExpectedFile) -> np.ndarray:
    """The synthetic native field of one file: `(members, rows, columns)`, distinct per member."""
    seed = file.step_minutes // 60 * 10 + len(file.field.variable)
    members = np.arange(len(REALIZATIONS), dtype=np.float32)[:, None, None] * 100
    cells = np.arange(N_ROWS * N_COLUMNS, dtype=np.float32).reshape(1, N_ROWS, N_COLUMNS)
    return members + cells / 7 + seed + 0.123456


def make_file(file: ExpectedFile, *, fill_cell: tuple[int, int] | None = None) -> bytes:
    """Build the HDF5 file a MOGREPS-G bucket would hold for `file`."""
    values = native_values(file).copy()
    if fill_cell is not None:
        values[:, fill_cell[0], fill_cell[1]] = 9.96921e36
    buffer = io.BytesIO()
    with h5py.File(buffer, "w") as h5:
        if file.field.level is None:
            data = values
            chunks = (1, CHUNK, CHUNK)
        else:
            heights = np.array([50.0, float(file.field.level), 150.0], dtype=np.float32)
            data = np.stack([values - 1000, values, values + 1000], axis=1)
            chunks = (1, 1, CHUNK, CHUNK)
            h5["height"] = heights
        h5.create_dataset(
            file.field.short_name, data=data, chunks=chunks, compression="gzip", compression_opts=1
        )
        h5["latitude"] = LATITUDE
        h5["longitude"] = LONGITUDE
        h5["realization"] = np.array(REALIZATIONS, dtype=np.int32)
        h5["forecast_period"] = np.int32(file.step_minutes * 60)
        h5["forecast_reference_time"] = np.int64(file.init_time.timestamp())
        h5["time"] = np.int64(file.init_time.timestamp() + file.step_minutes * 60)
        h5.attrs["um_version"] = b"13.8"
    return buffer.getvalue()


def expected_crop(file: ExpectedFile) -> np.ndarray:
    """What the archive should hold for one member of a file: the kept rectangle, flattened."""
    member = (file.member or 1) - 1
    native = native_values(file)[member]
    rows = slice(FIRST_ROW, FIRST_ROW + KEPT_ROWS)
    columns = slice(FIRST_COLUMN, FIRST_COLUMN + KEPT_COLUMNS)
    return native[rows, columns].ravel()


@dataclass
class FakeBucketG:
    """A stand-in for the Met Office MOGREPS-G bucket, serving synthetic files by byte range."""

    product: Product = TINY_MOGREPS_G
    requests: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Set up the source that lists the files, and the cache of built files."""
        self._source = MogrepsGSource(fetcher=Fetcher(client=httpx.Client()), base_url=BUCKET_URL)
        self._files: dict[str, ExpectedFile] = {}
        self._bodies: dict[str, bytes] = {}
        for file in self._source.expected_files(self.product, INIT):
            self._files[file.url] = file

    def handle(self, request: httpx.Request) -> httpx.Response:
        """Answer one request."""
        url = str(request.url)
        self.requests.append(url)
        file = self._files.get(url)
        if file is None:
            return httpx.Response(404)
        if url not in self._bodies:
            self._bodies[url] = make_file(file)
        body = self._bodies[url]
        span = re.fullmatch(r"bytes=(\d+)-(\d+)", request.headers.get("range", ""))
        if span is None:
            return httpx.Response(200, content=body)
        start, stop = int(span.group(1)), min(int(span.group(2)) + 1, len(body))
        return httpx.Response(
            206,
            content=body[start:stop],
            headers={"Content-Range": f"bytes {start}-{stop - 1}/{len(body)}"},
        )
