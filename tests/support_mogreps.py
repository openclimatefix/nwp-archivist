"""Test helpers for MOGREPS-UK: synthetic HDF5 files, a fake bucket that serves byte ranges."""

import io
import os
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import h5py
import httpx
import numpy as np
from support import Clock, FakeReporter

from nwp_archivist.dwd import Fetcher
from nwp_archivist.mogreps import LambertAzimuthalEqualArea, MogrepsSource, box_rectangle
from nwp_archivist.products import ExpectedFile, Field, Product
from nwp_archivist.recorder import Recorder, RecorderConfig
from nwp_archivist.source import Cropped, NotYet
from nwp_archivist.store import StoreLocation

BUCKET_URL = "https://bucket.test/uk-ensemble"
INIT = datetime(2026, 9, 23, 12, tzinfo=UTC)

# A small grid of 24 rows by 30 columns at 80 km (rows) and 50 km (columns) spacing, cut into 8 by
# 8 cell chunks, and placed so that the crop box keeps a rectangle inside it on every side.
N_ROWS = 24
N_COLUMNS = 30
CHUNK = 8
X_METRES = (np.arange(N_COLUMNS, dtype=np.float32) - 15) * 50_000
Y_METRES = (np.arange(N_ROWS, dtype=np.float32) - 12) * 80_000
PROJECTION = LambertAzimuthalEqualArea(
    latitude_origin=54.9, longitude_origin=-2.5, semi_major=6378137.0, semi_minor=6356752.31414036
)
GRID = box_rectangle(x=X_METRES, y=Y_METRES, projection=PROJECTION)
KEPT_ROWS, KEPT_COLUMNS = GRID.shape or (0, 0)
FIRST_ROW, FIRST_COLUMN = divmod(int(GRID.cell_index[0]), N_COLUMNS)
REALIZATIONS = (20, 21, 22)
HEIGHT_M = 100

TINY_MOGREPS = Product(
    name="tiny-mogreps",
    provider="test",
    licence="test",
    cycle_hours=1,
    n_members=3,
    fields=(
        Field("sw", "radiation_flux_in_shortwave_total_downward_at_surface", "sw_data", (60, 120)),
        Field("wind_100m", "wind_speed_on_height_levels", "wind_speed", (0, 60, 120), HEIGHT_M),
    ),
    hhl_levels=(),
    start_delay_hours=1.75,
    deadline_hours=24.0,
    source="mogreps",
    missing_after_hours=29 * 24.0,
    live_hours=6.0,
    has_realizations=True,
)


def native_values(file: ExpectedFile) -> np.ndarray:
    """The synthetic native field of one file: `(members, rows, columns)`, distinct per member."""
    seed = file.step_minutes // 60 * 10 + len(file.field.variable)
    members = np.arange(3, dtype=np.float32)[:, None, None] * 100
    cells = np.arange(N_ROWS * N_COLUMNS, dtype=np.float32).reshape(1, N_ROWS, N_COLUMNS)
    return members + cells / 7 + seed + 0.123456


def make_file(
    file: ExpectedFile,
    *,
    step_minutes: int | None = None,
    n_columns: int = N_COLUMNS,
    realizations: tuple[int, ...] = REALIZATIONS,
    fill_cell: tuple[int, int] | None = None,
) -> bytes:
    """Build the HDF5 file a MOGREPS-UK bucket would hold for `file`.

    Args:
        file: The file to build.
        step_minutes: The lead time the file claims, replaceable to build a mislabelled file.
        n_columns: The number of grid columns, replaceable to build a file of another grid.
        realizations: The provider's numbers for the three members.
        fill_cell: A `(row, column)` cell that holds netCDF's default fill value.

    Returns:
        The bytes of the file.
    """
    values = native_values(file)[:, :, :n_columns].copy()
    if fill_cell is not None:
        values[:, fill_cell[0], fill_cell[1]] = 9.96921e36
    lead = file.step_minutes if step_minutes is None else step_minutes
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
        h5["projection_x_coordinate"] = X_METRES[:n_columns]
        h5["projection_y_coordinate"] = Y_METRES
        h5["realization"] = np.array(realizations, dtype=np.int32)
        h5["forecast_period"] = np.int32(lead * 60)
        h5["forecast_reference_time"] = np.int64(file.init_time.timestamp())
        h5["time"] = np.int64(file.init_time.timestamp() + lead * 60)
        mapping = h5.create_dataset("lambert_azimuthal_equal_area", data=np.int32(-2147483647))
        mapping.attrs["grid_mapping_name"] = b"lambert_azimuthal_equal_area"
        mapping.attrs["latitude_of_projection_origin"] = np.array([54.9])
        mapping.attrs["longitude_of_projection_origin"] = np.array([-2.5])
        mapping.attrs["semi_major_axis"] = np.array([6378137.0])
        mapping.attrs["semi_minor_axis"] = np.array([6356752.31414036])
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
class FakeBucket:
    """A stand-in for the Met Office bucket, serving synthetic files with byte-range support.

    Attributes:
        product: The product whose files are served.
        published: Says whether a file is published yet.
        requests: Every URL requested, in order.
        overrides: Bodies that replace the normal one for a URL.
        kwargs: Arguments for `make_file`, to change every file the bucket builds.
    """

    product: Product = TINY_MOGREPS
    published: Callable[[ExpectedFile], bool] = lambda _file: True
    requests: list[str] = field(default_factory=list)
    overrides: dict[str, bytes] = field(default_factory=dict)
    kwargs: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Set up the source that lists the files, and the caches of built files."""
        self._source = MogrepsSource(fetcher=Fetcher(client=httpx.Client()), base_url=BUCKET_URL)
        self._files: dict[str, ExpectedFile] = {}
        self._bodies: dict[str, bytes] = {}

    def _index(self, init_time: datetime) -> None:
        for file in self._source.expected_files(self.product, init_time):
            self._files[file.url] = file

    def handle(self, request: httpx.Request) -> httpx.Response:
        """Answer one request."""
        url = str(request.url)
        self.requests.append(url)
        match = re.search(r"/(\d{4})/(\d\d)/(\d\d)/T(\d\d)00Z/", url)
        if match is None:
            return httpx.Response(404)
        init_time = datetime(*(int(part) for part in match.groups()), tzinfo=UTC)
        self._index(init_time)
        file = self._files.get(url)
        if file is None or not self.published(file):
            return httpx.Response(404)
        if url not in self._bodies:
            self._bodies[url] = self.overrides.get(url) or make_file(file, **self.kwargs)  # ty: ignore[invalid-argument-type]
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

    def urls_of_run(self, init_time: datetime) -> list[str]:
        """Every URL requested so far that belongs to the run."""
        marker = f"/{init_time:%Y/%m/%d}/T{init_time:%H%M}Z/"
        return [url for url in self.requests if marker in url]


def build_mogreps_recorder(
    tmp_path: Path,
    bucket: FakeBucket,
    clock: Clock,
    *,
    reporter: FakeReporter | None = None,
    worker_source_factory: Callable[[], Any] | None = None,
    **config_overrides: object,
) -> tuple[Recorder, FakeReporter]:
    """Build a recorder wired to the fake bucket, a local store, and a fake reporter."""
    reporter = reporter or FakeReporter()
    fetcher = Fetcher(
        client=httpx.Client(transport=httpx.MockTransport(bucket.handle)),
        sleep=lambda _seconds: None,
        jitter=lambda: 0.0,
    )
    config = RecorderConfig(
        store=StoreLocation(root=str(tmp_path / "store")),
        cache_dir=tmp_path / "cache",
        workers=2,
        backfill_workers=1,
        **config_overrides,  # ty: ignore[invalid-argument-type]
    )
    recorder = Recorder(
        config=config,
        fetcher=fetcher,
        reporter=reporter,
        clock=clock,
        mogreps_source=MogrepsSource(fetcher=fetcher, base_url=BUCKET_URL),
        **(
            {}
            if worker_source_factory is None
            else {"worker_source_factory": worker_source_factory}
        ),
    )
    return recorder, reporter


class FakeWorkerSource:
    """A source for worker processes that needs no network: it logs the process that fetched.

    The environment variable `FAKE_WORKER_LOG` names a file that gets one line per fetched file, and
    `FAKE_WORKER_FAIL`, `FAKE_WORKER_DIE` and `FAKE_WORKER_SLEEP` name a variable whose files raise
    an exception, kill the worker process, or hang for 60 seconds.
    """

    def fetch(self, file: ExpectedFile, grid: object) -> Cropped | NotYet:
        """Return the file's synthetic crop, or raise for the variable named to fail."""
        with open(os.environ["FAKE_WORKER_LOG"], "a") as log:  # noqa: PTH123
            log.write(f"{os.getpid()}\n")
        if file.field.variable == os.environ.get("FAKE_WORKER_DIE"):
            os._exit(1)
        if file.field.variable == os.environ.get("FAKE_WORKER_SLEEP"):
            time.sleep(60)
        if file.field.variable == os.environ.get("FAKE_WORKER_FAIL"):
            message = "the fake worker was told to fail"
            raise RuntimeError(message)
        return Cropped(
            expected_crop(file).astype(np.float32), N_ROWS * N_COLUMNS, 1308, REALIZATIONS
        )


def make_fake_worker_source() -> FakeWorkerSource:
    """Build the fake source in a worker process."""
    return FakeWorkerSource()


def hours_after(init_time: datetime, hours: float) -> datetime:
    """A time some hours after a run's initialisation time."""
    return init_time + timedelta(hours=hours)
