"""Test helpers: synthetic GRIB2 files, a tiny product, a fake provider, and a fake reporter."""

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

import eccodes
import httpx
import numpy as np

from nwp_archivist.dwd import Fetcher
from nwp_archivist.products import (
    ExpectedFile,
    Field,
    Product,
    expected_files,
    static_urls,
)
from nwp_archivist.recorder import Recorder, RecorderConfig
from nwp_archivist.reporting import FaultType
from nwp_archivist.store import StoreLocation

BASE_URL = "https://provider.test/v1/m"

# Six native cells. The last two are outside the crop box (one by 0.01 degrees), so a crop keeps
# the first four.
CLAT = np.array([50.0, 55.0, 61.5, 49.0, 61.51, 40.0], dtype=np.float32)
CLON = np.array([-10.0, 0.0, 3.5, 1.0, 1.0, 1.0], dtype=np.float32)
N_KEPT_CELLS = 4

TINY_EPS = Product(
    name="tiny-eps",
    provider="test",
    licence="test",
    cycle_hours=6,
    n_members=2,
    # The two variables have different step lists, so the padded step axis is exercised.
    fields=(
        Field("T_2M", "T_2M", "2t", (0, 60, 120)),
        Field("U_10M", "U_10M", "10u", (0, 60)),
    ),
    hhl_levels=(),
    start_delay_hours=1.0,
    deadline_hours=12.0,
)

TINY_DETERMINISTIC = Product(
    name="tiny-det",
    provider="test",
    licence="test",
    cycle_hours=6,
    n_members=None,
    fields=(Field("T_2M", "T_2M", "2t", (0, 60, 120)),),
    hhl_levels=(),
    start_delay_hours=1.0,
    deadline_hours=12.0,
)

INIT = datetime(2026, 9, 25, 6, tzinfo=UTC)


def make_grib(
    values: np.ndarray,
    *,
    short_name: str = "2t",
    step_minutes: int = 0,
    member: int | None = None,
    level: int | None = None,
) -> bytes:
    """Build one GRIB2 message. NaN entries are masked with a bitmap, as DWD masks cells."""
    handle = eccodes.codes_grib_new_from_samples("GRIB2")
    try:
        eccodes.codes_set(handle, "shortName", short_name)
        if level is not None:
            eccodes.codes_set(handle, "typeOfLevel", "generalVerticalLayer")
            eccodes.codes_set(handle, "level", level)
        if member is not None:
            eccodes.codes_set(handle, "productDefinitionTemplateNumber", 1)
            eccodes.codes_set(handle, "perturbationNumber", member)
        eccodes.codes_set(handle, "stepUnits", "m")
        eccodes.codes_set(handle, "endStep", step_minutes)
        eccodes.codes_set(handle, "packingType", "grid_ieee")
        masked = np.isnan(values)
        eccodes.codes_set(handle, "bitmapPresent", int(masked.any()))
        eccodes.codes_set(handle, "missingValue", 9999.0)
        eccodes.codes_set_values(handle, np.where(masked, 9999.0, values).astype(np.float64))
        return eccodes.codes_get_message(handle)
    finally:
        eccodes.codes_release(handle)


def data_values(file: ExpectedFile) -> np.ndarray:
    """The synthetic values of a data file, distinct for every variable, member, and step."""
    seed = (file.member or 0) * 100 + file.step_minutes // 60 * 10 + len(file.field.variable)
    return (np.arange(len(CLAT), dtype=np.float32) + 280.0 + seed).astype(np.float32)


@dataclass
class FakeProvider:
    """A stand-in for the DWD server, serving synthetic files for a set of products.

    Attributes:
        product: The product whose files are served.
        published: Says whether a data file is published yet.
        requests: Every URL requested, in order.
        overrides: Responses that replace the normal one for a URL.
        clat: The cell latitudes served, replaceable to simulate a grid change.
        clon: The cell longitudes served.
        static_available: Whether the static files are published yet.
    """

    product: Product
    published: Callable[[ExpectedFile], bool] = lambda _file: True
    requests: list[str] = field(default_factory=list)
    overrides: dict[str, httpx.Response] = field(default_factory=dict)
    clat: np.ndarray = field(default_factory=CLAT.copy)
    clon: np.ndarray = field(default_factory=CLON.copy)
    static_available: bool = True

    def __post_init__(self) -> None:
        """Index the files of the runs the tests use."""
        self._bodies: dict[str, bytes] = {}
        self._files: dict[str, ExpectedFile] = {}

    def _index_run(self, init_time: datetime) -> None:
        for file in expected_files(self.product, init_time, base_url=BASE_URL):
            self._files[file.url] = file

    def handle(self, request: httpx.Request) -> httpx.Response:
        """Answer one request."""
        url = str(request.url)
        self.requests.append(url)
        if url in self.overrides:
            return self.overrides[url]
        match = re.search(r"/r/(\d{4}-\d\d-\d\dT\d\d)%3A00", url)
        if not url.startswith(f"{BASE_URL}/{self.product.name}/") or match is None:
            return httpx.Response(404)
        init_time = datetime.strptime(match.group(1), "%Y-%m-%dT%H").replace(tzinfo=UTC)
        return self._answer(url, init_time)

    def _answer(self, url: str, init_time: datetime) -> httpx.Response:
        self._index_run(init_time)
        statics = static_urls(self.product, init_time, base_url=BASE_URL)
        for name, static_url in statics.items():
            if url == static_url:
                if not self.static_available:
                    return httpx.Response(404)
                values = {"clat": self.clat, "clon": self.clon}.get(name, CLAT * 0 + 1)
                return httpx.Response(200, content=make_grib(values, short_name="tlat"))
        file = self._files.get(url)
        if file is None or not self.published(file):
            return httpx.Response(404)
        body = make_grib(
            data_values(file),
            short_name=file.field.short_name,
            step_minutes=file.step_minutes,
            member=file.member,
        )
        return httpx.Response(200, content=body)


@dataclass
class FakeReporter:
    """Collects every fault and check-in instead of sending them."""

    faults: list[tuple[str, datetime | None, FaultType, str]] = field(default_factory=list)
    check_ins: list[bool] = field(default_factory=list)

    def fault(
        self, *, product: str, init_time: datetime | None, fault: FaultType, detail: str
    ) -> None:
        """Record one fault."""
        self.faults.append((product, init_time, fault, detail))

    def check_in(self, *, ok: bool) -> None:
        """Record one check-in."""
        self.check_ins.append(ok)

    def kinds(self) -> list[FaultType]:
        """The fault types reported so far, in order."""
        return [fault for _, _, fault, _ in self.faults]


class Clock:
    """A settable clock."""

    def __init__(self, now: datetime) -> None:
        """Start at `now`."""
        self.now = now

    def __call__(self) -> datetime:
        """The current fake time."""
        return self.now

    def advance(self, delta: timedelta) -> None:
        """Move time forward."""
        self.now += delta


def build_recorder(
    tmp_path: Path,
    provider: FakeProvider,
    clock: Clock,
    *,
    reporter: FakeReporter | None = None,
    location: StoreLocation | None = None,
    **config_overrides: object,
) -> tuple[Recorder, FakeReporter]:
    """Build a recorder wired to the fake provider, a local store, and a fake reporter."""
    reporter = reporter or FakeReporter()
    client = httpx.Client(transport=httpx.MockTransport(provider.handle))
    fetcher = Fetcher(client=client, sleep=lambda _seconds: None, jitter=lambda: 0.0)
    config = RecorderConfig(
        store=location or StoreLocation(root=str(tmp_path / "store")),
        cache_dir=tmp_path / "cache",
        workers=2,
        base_url=BASE_URL,
        **config_overrides,  # ty: ignore[invalid-argument-type]
    )
    return Recorder(config=config, fetcher=fetcher, reporter=reporter, clock=clock), reporter
