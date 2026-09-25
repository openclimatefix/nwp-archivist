"""Cropping, rounding, and the Icechunk repository of each product.

Each product is one Icechunk repository. Each variable is one Zarr array with dimensions
`(init_time, member, step, cell)`, or `(init_time, step, cell)` for a deterministic product, so that
a study opens one array per variable and slices it. Each run is written to the `init_time` slot
computed from its initialisation time, and one Icechunk commit per run is atomic across every
variable, so a reader never sees a half-written run.
"""

import os
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final
from urllib.parse import urlparse

import icechunk
import numpy as np
import zarr
import zarr.errors
from zarr.codecs import Crc32cCodec, ZstdCodec

from nwp_archivist.cache import RunGrid
from nwp_archivist.products import (
    BOX_LAT_MAX,
    BOX_LAT_MIN,
    BOX_LON_MAX,
    BOX_LON_MIN,
    Product,
)

# The first slot of every product's `init_time` axis. Every run's slot is its offset from here in
# whole cycles, so the axis is sorted whatever order the runs are committed in.
SLOT_EPOCH: Final[datetime] = datetime(2026, 1, 1, tzinfo=UTC)

# Manifest splitting keeps a commit's write volume flat as the repository grows. Without it, each
# commit rewrites the whole chunk-reference manifest of every array.
INIT_TIMES_PER_MANIFEST: Final[int] = 64

# Keep 13 of the 24 significand bits of each `float32` before writing, which makes the low 11 bits
# zero so that the compression codec finds repetition. The relative error is at most 2**-13.
SIGNIFICAND_BITS: Final[int] = 13

STATUS_COMPLETE: Final[int] = 1
STATUS_PARTIAL: Final[int] = 2
STATUS_MISSING: Final[int] = 3
STATUS_NAMES: Final[dict[int, str]] = {
    STATUS_COMPLETE: "complete",
    STATUS_PARTIAL: "partial",
    STATUS_MISSING: "missing",
}

# Status and coordinate arrays along `init_time` share one chunk size.
_INIT_TIME_CHUNK: Final[int] = 1024

_MAIN_BRANCH: Final[str] = "main"
_LAYOUT_VERSION: Final[int] = 1


def round_significand(values: np.ndarray, *, keep_bits: int = SIGNIFICAND_BITS) -> np.ndarray:
    """Round `float32` values to `keep_bits` significand bits, to nearest.

    This is Veltkamp splitting evaluated in `float32` (each operation rounds to nearest, and numpy
    never reassociates), the same method `delta_store.precision` uses in the forecasting project.
    NaN stays NaN.

    Args:
        values: A `float32` array.
        keep_bits: The number of significand bits to keep, from 2 to 22 inclusive (`float32` has 24
            in all, counting the implicit leading bit, and the splitting needs at least two bits of
            slack at each end).

    Returns:
        A new `float32` array of the same shape.
    """
    if values.dtype != np.float32:
        message = f"expected float32, got {values.dtype}"
        raise TypeError(message)
    shift = 24 - keep_bits
    if not 2 <= shift <= 22:
        message = f"keep_bits={keep_bits} is outside the range Veltkamp splitting supports"
        raise ValueError(message)
    splitter = np.float32(2**shift + 1)
    scaled = values * splitter
    return scaled - (scaled - values)


def box_cell_index(*, clat: np.ndarray, clon: np.ndarray) -> np.ndarray:
    """Find the native cells whose centre lies inside the crop box, edges included.

    Args:
        clat: The latitude of every native cell, in degrees north.
        clon: The longitude of every native cell, in degrees east.

    Returns:
        The indices of the cells inside the box, in increasing order, as `int32`.
    """
    inside = (
        (clat >= BOX_LAT_MIN)
        & (clat <= BOX_LAT_MAX)
        & (clon >= BOX_LON_MIN)
        & (clon <= BOX_LON_MAX)
    )
    return np.flatnonzero(inside).astype(np.int32)


def slot_for(product: Product, init_time: datetime) -> int:
    """The index of a run on the product's `init_time` axis."""
    offset = init_time - SLOT_EPOCH
    if offset % product.cycle:
        message = f"{init_time.isoformat()} is not on the {product.name} cycle"
        raise ValueError(message)
    return offset // product.cycle


def _init_time_seconds(product: Product, slots: range) -> np.ndarray:
    """The `init_time` coordinate values, in seconds since 1970, for a range of slots."""
    epoch_seconds = int(SLOT_EPOCH.timestamp())
    cycle_seconds = product.cycle_hours * 3600
    return np.array([epoch_seconds + slot * cycle_seconds for slot in slots], dtype=np.int64)


@dataclass(frozen=True)
class StoreLocation:
    """Where the product repositories live.

    Attributes:
        root: A local directory, or an `s3://bucket/prefix` address. Each product's repository is
            the `root/<product name>` directory or prefix.
        s3_endpoint_url: A custom S3 endpoint, used to test against a local S3 server. `None`
            means real AWS S3.
    """

    root: str
    s3_endpoint_url: str | None = None

    def storage(self, product_name: str) -> icechunk.Storage:
        """The Icechunk storage of one product's repository."""
        parsed = urlparse(self.root)
        if parsed.scheme != "s3":
            path = Path(self.root) / product_name
            path.mkdir(parents=True, exist_ok=True)
            return icechunk.local_filesystem_storage(str(path))
        prefix = "/".join(part for part in (parsed.path.strip("/"), product_name) if part)
        endpoint = self.s3_endpoint_url
        return icechunk.s3_storage(
            bucket=parsed.netloc,
            prefix=prefix,
            region=os.environ.get("AWS_DEFAULT_REGION", "eu-west-2"),
            endpoint_url=endpoint,
            allow_http=bool(endpoint and endpoint.startswith("http://")),
            force_path_style=endpoint is not None,
            from_env=True,
            # Source Cooperative requires the bucket owner to get full control of what we write.
            write_headers={"x-amz-acl": "bucket-owner-full-control"},
        )


def _repository_config() -> icechunk.RepositoryConfig:
    """The repository configuration, which every open must repeat because it is not persisted."""
    config = icechunk.RepositoryConfig.default()
    config.manifest = icechunk.ManifestConfig(
        splitting=icechunk.ManifestSplittingConfig.from_dict(
            {
                icechunk.ManifestSplitCondition.AnyArray(): {
                    icechunk.ManifestSplitDimCondition.DimensionName("init_time"): (
                        INIT_TIMES_PER_MANIFEST
                    )
                }
            }
        )
    )
    return config


@dataclass(frozen=True)
class RunToCommit:
    """Everything one Icechunk commit records about a run.

    Attributes:
        init_time: The run's initialisation time.
        status: `STATUS_COMPLETE`, `STATUS_PARTIAL` or `STATUS_MISSING`.
        files_expected: How many data files the run should contain.
        files_received: How many of them arrived.
        generating_process: The GRIB `generatingProcessIdentifier`, or 0 if no file arrived.
        archived_at: When the run was committed.
        code_version: The version of this package that recorded the run.
        load: Reads one cached file's cropped values, or `None` if it never arrived.
    """

    init_time: datetime
    status: int
    files_expected: int
    files_received: int
    generating_process: int
    archived_at: datetime
    code_version: str
    load: Callable[[str, int | None, int], np.ndarray | None]


@dataclass(frozen=True)
class StoredGrid:
    """The grid stored in a repository.

    Attributes:
        n_points: The number of cells on the provider's full native grid.
        cell_index: The indices of the archived cells within the full grid.
        clat: The latitude of the archived cells.
        clon: The longitude of the archived cells.
    """

    n_points: int
    cell_index: np.ndarray
    clat: np.ndarray
    clon: np.ndarray


class ProductStore:
    """The Icechunk repository of one product."""

    def __init__(self, *, repository: icechunk.Repository, product: Product) -> None:
        """Wrap an opened repository."""
        self.repository = repository
        self.product = product

    @classmethod
    def open(cls, *, location: StoreLocation, product: Product) -> ProductStore:
        """Open the product's repository, creating an empty one if there is none."""
        repository = icechunk.Repository.open_or_create(
            storage=location.storage(product.name), config=_repository_config()
        )
        return cls(repository=repository, product=product)

    def _read_group(self) -> zarr.Group | None:
        """Open the archive for reading, or return `None` while the repository is empty."""
        session = self.repository.readonly_session(branch=_MAIN_BRANCH)
        try:
            return zarr.open_group(session.store, mode="r")
        except zarr.errors.GroupNotFoundError:
            return None

    def has_layout(self) -> bool:
        """Whether the arrays exist yet, which they do once the first run's grid was stored."""
        group = self._read_group()
        return group is not None and "status" in group

    def statuses(self) -> np.ndarray:
        """The status of every slot: 0 for not archived, otherwise a `STATUS_*` code."""
        group = self._read_group()
        if group is None or "status" not in group:
            return np.zeros(0, dtype=np.int8)
        return np.asarray(_array(group, "status")[:], dtype=np.int8)

    def is_archived(self, init_time: datetime) -> bool:
        """Whether the repository holds a status for this run."""
        statuses = self.statuses()
        slot = slot_for(self.product, init_time)
        return slot < len(statuses) and statuses[slot] != 0

    def stored_grid(self) -> StoredGrid | None:
        """The stored grid, or `None` before the first run's grid was stored."""
        group = self._read_group()
        if group is None or "cell" not in group:
            return None
        return StoredGrid(
            n_points=int(group.attrs["n_points"]),  # ty: ignore[invalid-argument-type]
            cell_index=np.asarray(_array(group, "cell")[:]),
            clat=np.asarray(_array(group, "clat")[:]),
            clon=np.asarray(_array(group, "clon")[:]),
        )

    def layout_mismatch(self) -> str | None:
        """Describe how the stored layout differs from the product table, or return `None`."""
        group = self._read_group()
        if group is None or "status" not in group:
            return None
        stored = group.attrs.get("variables")
        if stored != self._variable_steps():
            return "the stored variables or step lists differ from the product table"
        if group.attrs.get("n_members") != self.product.n_members:
            return "the stored member count differs from the product table"
        return None

    def grid_mismatch(self, grid: RunGrid) -> str | None:
        """Describe how a run's grid differs from the stored grid, or return `None`."""
        stored = self.stored_grid()
        if stored is None:
            return None
        if grid.n_points != stored.n_points:
            return f"the native grid has {grid.n_points} cells, the archive holds {stored.n_points}"
        if not np.array_equal(grid.cell_index, stored.cell_index):
            return "the cells inside the crop box differ from the archive's"
        if not (
            np.array_equal(grid.statics["clat"], stored.clat)
            and np.array_equal(grid.statics["clon"], stored.clon)
        ):
            return "the cell coordinates differ from the archive's"
        return None

    def _variable_steps(self) -> dict[str, list[int]]:
        return {field.variable: list(field.steps_minutes) for field in self.product.fields}

    def _create_layout(self, grid: RunGrid) -> None:
        """Create every array, empty along `init_time`, and store the grid. Committed by caller."""
        product = self.product
        session = self.repository.writable_session(branch=_MAIN_BRANCH)
        group = zarr.open_group(session.store, mode="w")
        n_cells = len(grid.cell_index)
        n_steps = product.max_steps
        longest = max(product.fields, key=lambda field: len(field.steps_minutes))
        group.attrs.update(
            {
                "layout_version": _LAYOUT_VERSION,
                "product": product.name,
                "provider": product.provider,
                "licence": product.licence,
                "cycle_hours": product.cycle_hours,
                "n_members": product.n_members,
                "n_points": grid.n_points,
                "crop_box": {
                    "lat_min": BOX_LAT_MIN,
                    "lat_max": BOX_LAT_MAX,
                    "lon_min": BOX_LON_MIN,
                    "lon_max": BOX_LON_MAX,
                },
                "slot_epoch": SLOT_EPOCH.isoformat(),
                "status_codes": {str(code): name for code, name in STATUS_NAMES.items()},
                "variables": self._variable_steps(),
                "step_padding": (
                    "Each variable has its own lead times in step_of_<variable>, in minutes. "
                    "Entries beyond a variable's last lead time are -1, and the data there is NaN."
                ),
                "shortwave_note": (
                    "ASWDIR_S and ASWDIFD_S are averages since the start of the run, as delivered."
                ),
            }
        )
        codecs: tuple[Any, ...] = (ZstdCodec(level=3), Crc32cCodec())
        time_attrs = {
            "units": "seconds since 1970-01-01 00:00:00",
            "calendar": "proleptic_gregorian",
        }

        def along_init_time(name: str, dtype: str, attrs: dict[str, Any] | None = None) -> None:
            group.create_array(
                name,
                shape=(0,),
                chunks=(_INIT_TIME_CHUNK,),
                dtype=dtype,
                fill_value=0 if dtype != "str" else None,
                dimension_names=("init_time",),
                attributes=attrs or {},
            )

        along_init_time("init_time", "int64", time_attrs)
        along_init_time("status", "int8", {"description": "0 means not archived; see status_codes"})
        along_init_time("files_expected", "int32")
        along_init_time("files_received", "int32")
        along_init_time("archived_at", "int64", time_attrs)
        along_init_time("generating_process", "int32")
        along_init_time("code_version", "str")

        _write_static(group, "cell", grid.cell_index, dims=("cell",))
        _write_static(
            group,
            "step",
            np.array(longest.steps_minutes, dtype=np.int32),
            dims=("step",),
        )
        _array(group, "step").attrs["units"] = "minutes"
        if product.n_members is not None:
            _write_static(
                group,
                "member",
                np.array(product.members, dtype=np.int32),
                dims=("member",),
            )
        for name, values in grid.statics.items():
            _write_static(group, name, values.astype(np.float32), dims=("cell",))

        for field in product.fields:
            padded = np.full(n_steps, -1, dtype=np.int32)
            padded[: len(field.steps_minutes)] = field.steps_minutes
            _write_static(group, f"step_of_{field.variable}", padded, dims=("step",))
            _array(group, f"step_of_{field.variable}").attrs["units"] = "minutes"
            if product.n_members is None:
                shape: tuple[int, ...] = (0, n_steps, n_cells)
                chunks: tuple[int, ...] = (1, n_steps, n_cells)
                dims: tuple[str, ...] = ("init_time", "step", "cell")
            else:
                shape = (0, product.n_members, n_steps, n_cells)
                chunks = (1, 1, n_steps, n_cells)
                dims = ("init_time", "member", "step", "cell")
            group.create_array(
                field.variable,
                shape=shape,
                chunks=chunks,
                dtype="float32",
                fill_value=float("nan"),
                dimension_names=dims,
                compressors=codecs,
                attributes={
                    "dwd_parameter": field.parameter,
                    "model_level": field.level,
                    "step_coordinate": f"step_of_{field.variable}",
                },
            )
        session.commit(f"Create the {product.name} archive layout", allow_empty=True)

    def initialise(self, grid: RunGrid) -> None:
        """Create the layout from the first run's grid, unless the repository already has one."""
        if not self.has_layout():
            self._create_layout(grid)

    def commit_run(self, run: RunToCommit) -> str:
        """Write one run into its slot and commit it, in a single Icechunk commit.

        The run is built member by member from the cache, so each per-member chunk is written
        once. Writing the same run again overwrites the same slot.

        Args:
            run: The run's status, counts, and cached values.

        Returns:
            The identifier of the new snapshot.
        """
        product = self.product
        slot = slot_for(product, run.init_time)
        session = self.repository.writable_session(branch=_MAIN_BRANCH)
        group = zarr.open_group(session.store, mode="r+")
        self._grow(group, slot + 1)
        n_steps = product.max_steps
        n_cells = _array(group, "cell").shape[0]
        for field in product.fields:
            array = _array(group, field.variable)
            for member in product.members:
                block = np.full((n_steps, n_cells), np.nan, dtype=np.float32)
                for index, step in enumerate(field.steps_minutes):
                    values = run.load(field.variable, member, step)
                    if values is not None:
                        block[index] = values
                block = round_significand(block)
                if member is None:
                    array[slot] = block
                else:
                    array[slot, member - 1] = block
        _array(group, "status")[slot] = run.status
        _array(group, "files_expected")[slot] = run.files_expected
        _array(group, "files_received")[slot] = run.files_received
        _array(group, "archived_at")[slot] = int(run.archived_at.timestamp())
        _array(group, "generating_process")[slot] = run.generating_process
        _array(group, "code_version")[slot] = run.code_version
        message = (
            f"{product.name} {run.init_time:%Y-%m-%dT%H:%MZ} {STATUS_NAMES[run.status]} "
            f"{run.files_received}/{run.files_expected} files"
        )
        return session.commit(message, allow_empty=True)

    def _grow(self, group: zarr.Group, n_slots: int) -> None:
        """Extend every array along `init_time` to at least `n_slots`, filling the coordinate."""
        current = _array(group, "status").shape[0]
        if n_slots <= current:
            return
        for name in self._arrays_along_init_time(group):
            array = _array(group, name)
            array.resize((n_slots, *array.shape[1:]))
        _array(group, "init_time")[current:n_slots] = _init_time_seconds(
            self.product, range(current, n_slots)
        )

    def _arrays_along_init_time(self, group: zarr.Group) -> list[str]:
        names = [
            "init_time",
            "status",
            "files_expected",
            "files_received",
            "archived_at",
            "generating_process",
            "code_version",
        ]
        names.extend(field.variable for field in self.product.fields)
        return names


def _write_static(
    group: zarr.Group, name: str, values: np.ndarray, *, dims: tuple[str, ...]
) -> None:
    """Write a small array whole, as one chunk."""
    array = group.create_array(
        name,
        shape=values.shape,
        chunks=values.shape,
        dtype=values.dtype,
        dimension_names=dims,
        compressors=(ZstdCodec(level=3), Crc32cCodec()),
    )
    array[:] = values


def _array(group: zarr.Group, name: str) -> zarr.Array:
    """Fetch an array from a group, which the archive's layout guarantees exists."""
    member = group[name]
    if not isinstance(member, zarr.Array):
        message = f"{name} is not an array"
        raise TypeError(message)
    return member
