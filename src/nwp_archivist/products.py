"""The table of archived products and the exact list of files each run should contain.

Every other module reads this one table: the recorder computes which runs should exist and which
files each run holds, the store sizes its arrays, and the tests build their own small products from
the same dataclasses.
"""

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Final

DWD_BASE_URL: Final[str] = "https://opendata.dwd.de/weather/nwp/v1/m"
MOGREPS_BASE_URL: Final[str] = (
    "https://met-office-uk-ensemble-model-data.s3.eu-west-2.amazonaws.com/uk-ensemble"
)

# The "fat Great Britain" box, chosen so that it reaches offshore wind farms. It covers Northern
# Ireland, the Irish Sea, the Celtic Sea off Cornwall, the seas west of the Hebrides, Shetland, and
# the North Sea out to the Dutch and Belgian coasts. Native cells whose centre lies inside the box
# (edges included) are kept.
BOX_LAT_MIN: Final[float] = 49.0
BOX_LAT_MAX: Final[float] = 61.5
BOX_LON_MIN: Final[float] = -10.0
BOX_LON_MAX: Final[float] = 3.5

# The ICON vertical coordinate is a "generalVerticalLayer" (GRIB level type 150). Half-level heights
# (`HHL`) are published for the levels that bracket the layers we keep.
_MODEL_LEVEL_PATH: Final[str] = "lvt1/150/lv1"

# Static fields republished with every run, keyed by the name they get in the archive.
STATIC_PARAMETERS: Final[dict[str, str]] = {
    "clat": "CLAT",
    "clon": "CLON",
    "hsurf": "HSURF",
    "fr_land": "FR_LAND",
}


def _minutes(*segments: tuple[int, int, int]) -> tuple[int, ...]:
    """Build a lead-time list in minutes from `(first, last, every)` segments, ends included."""
    return tuple(
        minute for first, last, every in segments for minute in range(first, last + 1, every)
    )


_H: Final[int] = 60
_HOURLY_TO_48H: Final[tuple[int, ...]] = _minutes((0, 48 * _H, _H))
_EVERY_15_MIN_TO_48H: Final[tuple[int, ...]] = _minutes((0, 48 * _H, 15))
# ICON-EU-EPS surface fields are hourly to 75 h, then 3-hourly to 120 h (91 steps).
_EU_EPS_SURFACE_STEPS: Final[tuple[int, ...]] = _minutes(
    (0, 75 * _H, _H), (78 * _H, 120 * _H, 3 * _H)
)
# ICON-EU-EPS model-level fields are hourly to 51 h, then 6-hourly to 120 h (64 steps).
_EU_EPS_LEVEL_STEPS: Final[tuple[int, ...]] = _minutes(
    (0, 51 * _H, _H), (54 * _H, 120 * _H, 6 * _H)
)
# ICON-ART-EU fields come in three step lists (75, 64 and 89 steps).
_ART_STEPS_75: Final[tuple[int, ...]] = _minutes((0, 51 * _H, _H), (54 * _H, 120 * _H, 3 * _H))
_ART_STEPS_64: Final[tuple[int, ...]] = _minutes((0, 51 * _H, _H), (54 * _H, 120 * _H, 6 * _H))
_ART_STEPS_89: Final[tuple[int, ...]] = _minutes((0, 72 * _H, _H), (75 * _H, 120 * _H, 3 * _H))


@dataclass(frozen=True)
class Field:
    """One variable of a product: a DWD parameter, at one model level or at the surface.

    Attributes:
        variable: The name of the variable's array in the archive, such as `T_2M` or `U_L63`.
        parameter: The provider's name for the variable in the file's address, such as `T_2M` or
            `U` for DWD, or `temperature_at_screen_level` for MOGREPS-UK.
        short_name: The name the decoded file gives the variable (`shortName` in a DWD GRIB2
            message, the dataset name in a MOGREPS-UK file), which the decoder checks.
        steps_minutes: The lead times the provider publishes for this variable, in minutes.
        level: The DWD model level, or the height in metres of a MOGREPS-UK height-level field,
            or `None` for a surface field.
    """

    variable: str
    parameter: str
    short_name: str
    steps_minutes: tuple[int, ...]
    level: int | None = None


@dataclass(frozen=True)
class ExpectedFile:
    """One file a run should contain.

    Attributes:
        field: The variable this file belongs to.
        member: The ensemble member number (from 1), or `None` for a deterministic product.
        step_minutes: The lead time in minutes.
        url: The address of the file.
        init_time: The initialisation time of the run the file belongs to.
    """

    field: Field
    member: int | None
    step_minutes: int
    url: str
    init_time: datetime


@dataclass(frozen=True)
class Product:
    """A weather product archived as one Icechunk repository.

    Attributes:
        name: The product name, which is also the DWD directory name and the repository name.
        provider: The organisation that publishes the product.
        licence: The licence of the published data.
        cycle_hours: The hours between two runs.
        n_members: The number of ensemble members, or `None` for a deterministic product.
        fields: The variables archived for every run.
        hhl_levels: The model levels whose half-level heights (`HHL`) are archived, if any.
        start_delay_hours: How long after the initialisation time the recorder starts fetching.
        deadline_hours: How long after the initialisation time the recorder commits whatever has
            arrived. It is set from how long the provider keeps a run.
        source: Which provider's files these are: `dwd` or `mogreps`.
        lookback_hours: How far back from now a cycle looks for runs that are not yet archived,
            or `None` for the recorder's default.
        missing_after_hours: How long after the initialisation time a run with no file at all is
            recorded as `missing`, or `None` for the same as `deadline_hours`.
        live_hours: Runs younger than this are live and are handled first; older runs are a
            backfill that the recorder handles newest first, within a time budget. `None` means
            every run is live.
        cell_chunk: The number of cells in one Zarr chunk, or `None` for all cells in one chunk.
        has_realizations: Whether the provider labels each member of a run with a realization
            number that changes from run to run, which the archive then stores.
        notes: Attributes written to the archive's root group, saying what the fields mean.
    """

    name: str
    provider: str
    licence: str
    cycle_hours: int
    n_members: int | None
    fields: tuple[Field, ...]
    hhl_levels: tuple[int, ...]
    start_delay_hours: float
    deadline_hours: float
    source: str = "dwd"
    lookback_hours: float | None = None
    missing_after_hours: float | None = None
    live_hours: float | None = None
    cell_chunk: int | None = None
    has_realizations: bool = False
    notes: tuple[tuple[str, str], ...] = ()

    @property
    def members(self) -> tuple[int | None, ...]:
        """The member numbers, or `(None,)` for a deterministic product."""
        if self.n_members is None:
            return (None,)
        return tuple(range(1, self.n_members + 1))

    def __post_init__(self) -> None:
        """Check that every field's lead times are a subset of the longest field's."""
        axis = set(self.step_axis)
        for field in self.fields:
            if not set(field.steps_minutes) <= axis:
                message = f"{field.variable} has lead times outside {self.name}'s step axis"
                raise ValueError(message)

    @property
    def step_axis(self) -> tuple[int, ...]:
        """The product's `step` axis: the lead times of its longest field, in minutes."""
        return max((field.steps_minutes for field in self.fields), key=len)

    @property
    def max_steps(self) -> int:
        """The length of the `step` axis."""
        return len(self.step_axis)

    def step_index(self, step_minutes: int) -> int:
        """The position of a lead time on the `step` axis."""
        return self.step_axis.index(step_minutes)

    @property
    def cycle(self) -> timedelta:
        """The time between two runs."""
        return timedelta(hours=self.cycle_hours)

    def deadline(self, init_time: datetime) -> datetime:
        """The time after which the recorder commits whatever has arrived for a run."""
        return init_time + timedelta(hours=self.deadline_hours)

    def missing_deadline(self, init_time: datetime) -> datetime:
        """The time after which a run with no file at all is recorded as `missing`."""
        return init_time + timedelta(hours=self.missing_after_hours or self.deadline_hours)

    def field_by_variable(self, variable: str) -> Field:
        """Look a variable up by its archive name."""
        for field in self.fields:
            if field.variable == variable:
                return field
        raise KeyError(variable)


def run_directory(init_time: datetime) -> str:
    """Format an initialisation time as DWD's run directory name, `YYYY-MM-DDTHH%3A00`."""
    return f"{init_time:%Y-%m-%dT%H}%3A00"


def step_file_name(step_minutes: int) -> str:
    """Format a lead time as DWD's file name, such as `PT024H15M.grib2`."""
    return f"PT{step_minutes // 60:03d}H{step_minutes % 60:02d}M.grib2"


def _url(
    *,
    base_url: str,
    product: Product,
    parameter: str,
    level: int | None,
    init_time: datetime,
    member: int | None,
    step_minutes: int,
) -> str:
    """Assemble one file's address following DWD's open-data directory layout."""
    level_path = "" if level is None else f"/{_MODEL_LEVEL_PATH}/{level}"
    member_path = "" if member is None else f"/e/{member:02d}"
    return (
        f"{base_url}/{product.name}/p/{parameter}{level_path}/r/{run_directory(init_time)}"
        f"{member_path}/s/{step_file_name(step_minutes)}"
    )


def expected_files(
    product: Product, init_time: datetime, *, base_url: str = DWD_BASE_URL
) -> list[ExpectedFile]:
    """List every file a run of `product` should contain, in the order they are published.

    The order is by variable, then member, then lead time, so that a fetcher can stop at the first
    file that is not yet published in each sequence.

    Args:
        product: The product to list files for.
        init_time: The run's initialisation time.
        base_url: The root of the directory layout, replaceable so that tests can point elsewhere.

    Returns:
        One `ExpectedFile` for each variable, member and lead time.
    """
    return [
        ExpectedFile(
            field=field,
            member=member,
            step_minutes=step,
            url=_url(
                base_url=base_url,
                product=product,
                parameter=field.parameter,
                level=field.level,
                init_time=init_time,
                member=member,
                step_minutes=step,
            ),
            init_time=init_time,
        )
        for field in product.fields
        for member in product.members
        for step in field.steps_minutes
    ]


def static_urls(
    product: Product, init_time: datetime, *, base_url: str = DWD_BASE_URL
) -> dict[str, str]:
    """List the addresses of the static fields of a run, keyed by their archive name.

    The static fields are republished with every run (and with every member) as the file for lead
    time zero. Ensembles are read from member 1.

    Args:
        product: The product to list files for.
        init_time: The run's initialisation time.
        base_url: The root of the directory layout.

    Returns:
        The address of `clat`, `clon`, `hsurf`, `fr_land` and one `hhl_L{level}` per half level.
    """
    member = None if product.n_members is None else 1

    def url(parameter: str, level: int | None) -> str:
        return _url(
            base_url=base_url,
            product=product,
            parameter=parameter,
            level=level,
            init_time=init_time,
            member=member,
            step_minutes=0,
        )

    urls = {name: url(parameter, None) for name, parameter in STATIC_PARAMETERS.items()}
    urls.update({f"hhl_L{level}": url("HHL", level) for level in product.hhl_levels})
    return urls


def run_init_times(product: Product, *, first: datetime, last: datetime) -> list[datetime]:
    """List the run initialisation times of `product` in the closed interval `[first, last]`."""
    day = first.astimezone(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
    candidate = day
    times: list[datetime] = []
    while candidate <= last:
        if candidate >= first:
            times.append(candidate)
        candidate += product.cycle
    return times


_DWD_NOTES: Final[tuple[tuple[str, str], ...]] = (
    (
        "shortwave_note",
        "ASWDIR_S and ASWDIFD_S are averages since the start of the run, as delivered.",
    ),
)


def _surface_fields(steps: tuple[int, ...], radiation_steps: tuple[int, ...]) -> tuple[Field, ...]:
    """The six fields every DWD product archives: shortwave radiation, wind, temperature, cloud."""
    return (
        Field("ASWDIR_S", "ASWDIR_S", "ASWDIR_S", radiation_steps),
        Field("ASWDIFD_S", "ASWDIFD_S", "ASWDIFD_S", radiation_steps),
        Field("T_2M", "T_2M", "2t", steps),
        Field("U_10M", "U_10M", "10u", steps),
        Field("V_10M", "V_10M", "10v", steps),
        Field("CLCT", "CLCT", "CLCT", steps),
    )


def _wind_at_levels(levels: tuple[int, ...], steps: tuple[int, ...]) -> tuple[Field, ...]:
    """Wind components on model levels."""
    return tuple(
        Field(f"{name.upper()}_L{level}", name.upper(), name, steps, level)
        for level in levels
        for name in ("u", "v")
    )


ICON_D2_EPS: Final[Product] = Product(
    name="icon-d2-eps",
    provider="DWD",
    licence="CC BY 4.0",
    cycle_hours=3,
    n_members=20,
    # Model levels 63 and 62 are centred at about 77 m and 125 m, so 100 m lies between them.
    fields=_surface_fields(_HOURLY_TO_48H, _HOURLY_TO_48H)
    + _wind_at_levels((63, 62), _HOURLY_TO_48H),
    hhl_levels=(62, 63, 64),
    start_delay_hours=0.5,
    deadline_hours=23.0,
    notes=_DWD_NOTES,
)

ICON_D2: Final[Product] = Product(
    name="icon-d2",
    provider="DWD",
    licence="CC BY 4.0",
    cycle_hours=3,
    n_members=None,
    # The deterministic run publishes shortwave radiation every 15 minutes.
    fields=_surface_fields(_HOURLY_TO_48H, _EVERY_15_MIN_TO_48H)
    + _wind_at_levels((63, 62), _HOURLY_TO_48H),
    hhl_levels=(62, 63, 64),
    start_delay_hours=0.5,
    deadline_hours=23.0,
    notes=_DWD_NOTES,
)

ICON_EU_EPS: Final[Product] = Product(
    name="icon-eu-eps",
    provider="DWD",
    licence="CC BY 4.0",
    cycle_hours=6,
    n_members=40,
    # Model level 72 is centred at about 95 m and stands in for 100 m.
    fields=_surface_fields(_EU_EPS_SURFACE_STEPS, _EU_EPS_SURFACE_STEPS)
    + _wind_at_levels((72,), _EU_EPS_LEVEL_STEPS),
    hhl_levels=(72, 73, 74),
    start_delay_hours=2.0,
    deadline_hours=24.0,
    notes=_DWD_NOTES,
)

ICON_ART_EU: Final[Product] = Product(
    name="icon-art-eu",
    provider="DWD",
    licence="CC BY 4.0",
    cycle_hours=6,
    n_members=None,
    fields=(
        Field("ASWDIR_S", "ASWDIR_S", "ASWDIR_S", _ART_STEPS_75),
        Field("ASWDIFD_S", "ASWDIFD_S", "ASWDIFD_S", _ART_STEPS_75),
        Field("T_2M", "T_2M", "2t", _ART_STEPS_75),
        Field("CLCT", "CLCT", "CLCT", _ART_STEPS_75),
        Field("ASOB_S_CS", "ASOB_S_CS", "avg_snswrfcs", _ART_STEPS_64),
        Field("TAOD_DUST", "TAOD_DUST", "TAOD_DUST", _ART_STEPS_89),
    ),
    hhl_levels=(),
    start_delay_hours=2.0,
    deadline_hours=24.0,
    notes=_DWD_NOTES,
)

# MOGREPS-UK publishes 15-minute lead times to 11 h 45 min and hourly lead times after that, for
# 9 of the 10 fields we keep, and hourly lead times only for the rest. Shortwave has no lead time 0.
_MOGREPS_HOURLY: Final[tuple[int, ...]] = _minutes((0, 126 * _H, _H))
_MOGREPS_HOURLY_FROM_1H: Final[tuple[int, ...]] = _minutes((_H, 126 * _H, _H))
_MOGREPS_MIXED: Final[tuple[int, ...]] = _minutes((0, 705, 15), (12 * _H, 126 * _H, _H))
_MOGREPS_FILE: Final[str] = "radiation_flux_in_shortwave_{}_downward_at_surface"
_MOGREPS_HEIGHT_M: Final[int] = 100

MOGREPS_UK: Final[Product] = Product(
    name="mogreps-uk",
    provider="Met Office",
    licence="CC BY-SA 4.0",
    cycle_hours=1,
    n_members=3,
    fields=(
        Field(
            "shortwave_total",
            _MOGREPS_FILE.format("total"),
            "surface_downwelling_shortwave_flux_in_air",
            _MOGREPS_HOURLY_FROM_1H,
        ),
        Field(
            "shortwave_direct",
            _MOGREPS_FILE.format("direct"),
            "surface_direct_downwelling_shortwave_flux_in_air",
            _MOGREPS_HOURLY_FROM_1H,
        ),
        Field(
            "shortwave_diffuse",
            _MOGREPS_FILE.format("diffuse"),
            "surface_diffusive_downwelling_shortwave_flux_in_air",
            _MOGREPS_HOURLY_FROM_1H,
        ),
        Field("temperature_1p5m", "temperature_at_screen_level", "air_temperature", _MOGREPS_MIXED),
        Field("wind_speed_10m", "wind_speed_at_10m", "wind_speed", _MOGREPS_MIXED),
        Field("wind_direction_10m", "wind_direction_at_10m", "wind_from_direction", _MOGREPS_MIXED),
        Field("cloud_total", "cloud_amount_of_total_cloud", "cloud_area_fraction", _MOGREPS_HOURLY),
        Field(
            "wind_speed_100m",
            "wind_speed_on_height_levels",
            "wind_speed",
            _MOGREPS_HOURLY,
            _MOGREPS_HEIGHT_M,
        ),
        Field(
            "wind_direction_100m",
            "wind_direction_on_height_levels",
            "wind_from_direction",
            _MOGREPS_HOURLY,
            _MOGREPS_HEIGHT_M,
        ),
    ),
    hhl_levels=(),
    # The first file of a run appears about 1 h 44 min after initialisation and the last about
    # 2 h 38 min after.
    start_delay_hours=1.75,
    deadline_hours=24.0,
    source="mogreps",
    # The bucket deletes an object 30 days after it was written (rounded up to midnight UTC). A run
    # with no file is `missing` only one day before its files could no longer be fetched.
    lookback_hours=30 * 24.0,
    missing_after_hours=29 * 24.0,
    live_hours=6.0,
    cell_chunk=44_000,
    has_realizations=True,
    notes=(
        (
            "shortwave_note",
            (
                "The shortwave files carry no cell_methods and no time bounds, and their time is "
                "the valid time, so the values are taken to be instantaneous. Stored as delivered."
            ),
        ),
        (
            "generating_process_note",
            "The Unified Model version from the file (13.8 is stored as 1308).",
        ),
        (
            "grid_note",
            (
                "The cells are the rectangle of the Met Office 2 km Lambert azimuthal equal-area "
                "grid that contains the crop box, flattened row by row; grid_shape is (rows, "
                "columns). Members are numbered 1 to 3 in file order; the realization array holds "
                "the Met Office's number for each, which changes from run to run."
            ),
        ),
    ),
)

PRODUCTS: Final[dict[str, Product]] = {
    product.name: product
    for product in (ICON_D2_EPS, ICON_D2, ICON_EU_EPS, ICON_ART_EU, MOGREPS_UK)
}
