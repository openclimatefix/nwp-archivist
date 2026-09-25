from datetime import datetime, timedelta
from pathlib import Path

import httpx
import numpy as np
import pytest
import zarr
from support import Clock
from support_mogreps import (
    BUCKET_URL,
    CHUNK,
    FIRST_COLUMN,
    FIRST_ROW,
    GRID,
    INIT,
    KEPT_COLUMNS,
    KEPT_ROWS,
    N_COLUMNS,
    N_ROWS,
    PROJECTION,
    REALIZATIONS,
    TINY_MOGREPS,
    X_METRES,
    Y_METRES,
    FakeBucket,
    build_mogreps_recorder,
    expected_crop,
    hours_after,
    make_file,
)

from nwp_archivist import mogreps as mogreps_module
from nwp_archivist import recorder as recorder_module
from nwp_archivist.cache import ProductCache
from nwp_archivist.dwd import Fetcher
from nwp_archivist.mogreps import LambertAzimuthalEqualArea, MogrepsSource, box_rectangle
from nwp_archivist.products import BOX_LAT_MAX, BOX_LAT_MIN, MOGREPS_UK, ExpectedFile
from nwp_archivist.reporting import FaultType
from nwp_archivist.source import Cropped, NotYet
from nwp_archivist.store import (
    STATUS_COMPLETE,
    STATUS_MISSING,
    STATUS_PARTIAL,
    ProductStore,
    StoreLocation,
    slot_for,
)

REAL_PROJECTION = LambertAzimuthalEqualArea(
    latitude_origin=54.9, longitude_origin=-2.5, semi_major=6378137.0, semi_minor=6356752.31414036
)


def _source(bucket: FakeBucket) -> MogrepsSource:
    client = httpx.Client(transport=httpx.MockTransport(bucket.handle))
    fetcher = Fetcher(client=client, sleep=lambda _seconds: None, jitter=lambda: 0.0)
    return MogrepsSource(fetcher=fetcher, base_url=BUCKET_URL)


def _file(variable: str, step: int, member: int) -> ExpectedFile:
    (file,) = [
        f
        for f in _source(FakeBucket()).expected_files(TINY_MOGREPS, INIT)
        if f.field.variable == variable and f.step_minutes == step and f.member == member
    ]
    return file


# --- products and addresses -------------------------------------------------------------------


def test_a_mogreps_run_lists_one_file_per_member_variable_and_lead_time() -> None:
    files = MogrepsSource(fetcher=Fetcher(client=httpx.Client())).expected_files(MOGREPS_UK, INIT)
    counts = {f.field.variable: 0 for f in files}
    for file in files:
        counts[file.field.variable] += 1
    assert counts == {
        "shortwave_total": 3 * 126,
        "shortwave_direct": 3 * 126,
        "shortwave_diffuse": 3 * 126,
        "temperature_1p5m": 3 * 163,
        "wind_speed_10m": 3 * 163,
        "wind_direction_10m": 3 * 163,
        "cloud_total": 3 * 127,
        "wind_speed_100m": 3 * 127,
        "wind_direction_100m": 3 * 127,
    }
    assert MOGREPS_UK.max_steps == 163


def test_the_address_of_a_file_follows_the_bucket_layout() -> None:
    files = MogrepsSource(fetcher=Fetcher(client=httpx.Client())).expected_files(MOGREPS_UK, INIT)
    urls = {(f.field.variable, f.step_minutes): f.url for f in files if f.member == 1}
    assert urls[("wind_speed_100m", 7560)].startswith(
        "https://met-office-uk-ensemble-model-data.s3.eu-west-2.amazonaws.com/uk-ensemble/"
    )
    assert urls[("wind_speed_100m", 7560)].endswith(
        "/uk-ensemble/2026/09/23/T1200Z/20260928T1800Z-PT0126H00M-wind_speed_on_height_levels.nc"
    )
    assert urls[("temperature_1p5m", 15)].endswith(
        "/2026/09/23/T1200Z/20260923T1215Z-PT0000H15M-temperature_at_screen_level.nc"
    )
    assert ("shortwave_total", 0) not in urls


def test_temperature_has_15_minute_lead_times_for_the_first_12_hours_only() -> None:
    steps = MOGREPS_UK.field_by_variable("temperature_1p5m").steps_minutes
    assert steps[:4] == (0, 15, 30, 45)
    assert steps[47:50] == (705, 720, 780)
    assert steps[-1] == 126 * 60


# --- projection and crop ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("x", "y", "lon", "lat"),
    [
        (0.0, 0.0, -2.5, 54.9),
        (-1158000.0, -1036000.0, -17.117129281713446, 44.51715281686621),
        (924000.0, 902000.0, 15.279769222349213, 61.92068680395195),
        (100000.0, 200000.0, -0.8683151638684354, 56.685934735985256),
        (-500000.0, 300000.0, -10.820306732755775, 57.32595848960826),
    ],
)
def test_the_projection_inverse_matches_proj(x: float, y: float, lon: float, lat: float) -> None:
    # The reference values are `pyproj` output for the file's ellipsoid and projection origin.
    got_lon, got_lat = REAL_PROJECTION.inverse(np.array(x), np.array(y))
    assert float(got_lon) == pytest.approx(lon, abs=1e-8)
    assert float(got_lat) == pytest.approx(lat, abs=1e-7)


def test_the_crop_box_on_the_real_grid_is_707_rows_by_494_columns() -> None:
    x = np.arange(1042, dtype=np.float32) * 2000 - 1158000
    y = np.arange(970, dtype=np.float32) * 2000 - 1036000
    grid = box_rectangle(x=x, y=y, projection=REAL_PROJECTION)
    assert grid.shape == (707, 494)
    assert grid.n_points == 970 * 1042
    assert len(grid.cell_index) == 707 * 494
    assert divmod(int(grid.cell_index[0]), 1042) == (190, 305)
    assert grid.statics["clat"].dtype == np.float32


def test_every_cell_inside_the_box_is_kept_and_the_rectangle_is_the_smallest() -> None:
    lon, lat = PROJECTION.inverse(*np.meshgrid(X_METRES.astype(float), Y_METRES.astype(float)))
    inside = (lat >= BOX_LAT_MIN) & (lat <= BOX_LAT_MAX) & (lon >= -10.0) & (lon <= 3.5)
    kept = np.zeros(inside.shape, dtype=bool)
    kept.ravel()[GRID.cell_index] = True
    assert kept[inside].all()
    rows, columns = np.nonzero(inside)
    assert GRID.shape == (rows.max() - rows.min() + 1, columns.max() - columns.min() + 1)
    assert GRID.shape == (KEPT_ROWS, KEPT_COLUMNS)
    assert 0 < FIRST_ROW < FIRST_ROW + KEPT_ROWS < N_ROWS
    assert 0 < FIRST_COLUMN < FIRST_COLUMN + KEPT_COLUMNS < N_COLUMNS


# --- reading one file -------------------------------------------------------------------------


@pytest.mark.parametrize("variable", ["sw", "wind_100m"])
@pytest.mark.parametrize("member", [1, 2, 3])
def test_a_fetched_member_is_the_kept_rectangle_of_its_own_field(
    variable: str, member: int
) -> None:
    step = 60
    file = _file(variable, step, member)
    result = _source(FakeBucket()).fetch(file, GRID)
    assert isinstance(result, Cropped)
    np.testing.assert_array_equal(result.values, expected_crop(file).astype(np.float32))
    assert result.n_points == N_ROWS * N_COLUMNS
    assert result.member_ids == REALIZATIONS
    assert result.generating_process == 1308


def test_a_height_level_file_is_read_at_the_level_the_field_names() -> None:
    file = _file("wind_100m", 0, 1)
    result = _source(FakeBucket()).fetch(file, GRID)
    assert isinstance(result, Cropped)
    # The synthetic file holds the field minus 1000 at the level below 100 m and plus 1000 above.
    np.testing.assert_array_equal(result.values, expected_crop(file).astype(np.float32))


def test_the_chunks_are_downloaded_in_a_few_range_requests_not_one_per_chunk() -> None:
    bucket = FakeBucket()
    _source(bucket).fetch(_file("sw", 60, 1), GRID)
    chunks = -(-KEPT_ROWS // CHUNK + 1) * -(-KEPT_COLUMNS // CHUNK + 1)
    assert 0 < len(bucket.requests) < chunks


def test_a_cell_holding_the_netcdf_fill_value_is_nan() -> None:
    file = _file("sw", 60, 1)
    row, column = FIRST_ROW + 2, FIRST_COLUMN + 3
    bucket = FakeBucket(overrides={file.url: make_file(file, fill_cell=(row, column))})
    result = _source(bucket).fetch(file, GRID)
    assert isinstance(result, Cropped)
    assert np.isnan(result.values[2 * KEPT_COLUMNS + 3])
    assert np.isnan(result.values).sum() == 1


def test_an_unpublished_file_is_not_yet_with_reason_404() -> None:
    result = _source(FakeBucket(published=lambda _file: False)).fetch(_file("sw", 60, 1), GRID)
    assert result == NotYet("404")


def test_a_file_that_claims_another_lead_time_is_not_yet() -> None:
    file = _file("sw", 60, 1)
    bucket = FakeBucket(overrides={file.url: make_file(file, step_minutes=120)})
    result = _source(bucket).fetch(file, GRID)
    assert isinstance(result, NotYet)
    assert "lead time" in result.reason


def test_a_file_of_another_run_is_not_yet() -> None:
    file = _file("sw", 60, 1)
    other = ExpectedFile(
        field=file.field,
        member=1,
        step_minutes=60,
        url="",
        init_time=hours_after(INIT, -1),
    )
    bucket = FakeBucket(overrides={file.url: make_file(other)})
    result = _source(bucket).fetch(file, GRID)
    assert isinstance(result, NotYet)
    assert "run" in result.reason


def test_a_file_without_the_named_height_is_not_yet() -> None:
    file = _file("wind_100m", 0, 1)
    other_field = ExpectedFile(
        field=type(file.field)("wind_100m", "x", "wind_speed", (0,), 999),
        member=1,
        step_minutes=0,
        url="",
        init_time=INIT,
    )
    bucket = FakeBucket(overrides={file.url: make_file(other_field)})
    result = _source(bucket).fetch(file, GRID)
    assert isinstance(result, NotYet)
    assert "100 m" in result.reason


def test_a_truncated_body_is_not_yet_and_a_later_good_body_is_read() -> None:
    file = _file("sw", 60, 1)
    good = make_file(file)
    bucket = FakeBucket(overrides={file.url: good[: len(good) // 2]})
    source = _source(bucket)
    assert isinstance(source.fetch(file, GRID), NotYet)
    bucket.overrides[file.url] = good
    bucket._bodies.clear()
    assert isinstance(source.fetch(file, GRID), Cropped)


def test_garbage_is_not_yet_and_never_raises() -> None:
    file = _file("sw", 60, 1)
    bucket = FakeBucket(overrides={file.url: b"not an hdf5 file" * 100})
    assert isinstance(_source(bucket).fetch(file, GRID), NotYet)


def test_a_chunk_that_cannot_be_decompressed_is_not_yet_and_the_next_fetch_starts_afresh(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source(FakeBucket())
    with monkeypatch.context() as patched:

        def broken(_data: bytes) -> bytes:
            message = "bad chunk"
            raise mogreps_module.zlib.error(message)

        patched.setattr(mogreps_module.zlib, "decompress", broken)
        result = source.fetch(_file("sw", 60, 1), GRID)
    assert isinstance(result, NotYet)
    assert "bad chunk" in result.reason
    assert isinstance(source.fetch(_file("sw", 60, 1), GRID), Cropped)


def test_a_file_on_a_different_grid_reports_its_own_cell_count() -> None:
    file = _file("sw", 60, 1)
    bucket = FakeBucket(overrides={file.url: make_file(file, n_columns=N_COLUMNS - 2)})
    result = _source(bucket).fetch(file, GRID)
    assert isinstance(result, Cropped)
    assert result.n_points == N_ROWS * (N_COLUMNS - 2)


def test_the_grid_comes_from_a_lead_time_zero_file() -> None:
    bucket = FakeBucket()
    grid = _source(bucket).fetch_grid(
        product=TINY_MOGREPS, init_time=INIT, only=None, previous=None
    )
    assert not isinstance(grid, NotYet)
    np.testing.assert_array_equal(grid.cell_index, GRID.cell_index)
    assert grid.shape == GRID.shape
    assert all("PT0000H00M" in url for url in bucket.requests)


def test_the_grid_is_not_yet_while_the_lead_time_zero_file_is_unpublished() -> None:
    bucket = FakeBucket(published=lambda _file: False)
    grid = _source(bucket).fetch_grid(
        product=TINY_MOGREPS, init_time=INIT, only=None, previous=None
    )
    assert grid == NotYet("404")


def test_the_members_of_one_file_share_one_open_file(monkeypatch: pytest.MonkeyPatch) -> None:
    bucket = FakeBucket()
    source = _source(bucket)
    opened: list[str] = []
    original = mogreps_module.h5py.File

    def counting(*args: object, **kwargs: object) -> object:
        if args[1] == "r":
            opened.append("open")
        return original(*args, **kwargs)

    monkeypatch.setattr(mogreps_module.h5py, "File", counting)
    for member in (1, 2, 3):
        assert isinstance(source.fetch(_file("sw", 60, member), GRID), Cropped)
    assert opened == ["open"]


# --- the recorder with MOGREPS-UK -------------------------------------------------------------

SOON = hours_after(INIT, 2.5)


def _store(tmp_path: Path) -> ProductStore:
    return ProductStore.open(
        location=StoreLocation(root=str(tmp_path / "store")), product=TINY_MOGREPS
    )


def _status(tmp_path: Path, init_time: datetime = INIT) -> int:
    statuses = _store(tmp_path).statuses()
    slot = slot_for(TINY_MOGREPS, init_time)
    return int(statuses[slot]) if len(statuses) > slot else 0


def _group(tmp_path: Path) -> zarr.Group:
    session = _store(tmp_path).repository.readonly_session(branch="main")
    return zarr.open_group(session.store, mode="r")


def _stored(tmp_path: Path, name: str) -> np.ndarray:
    array = _group(tmp_path)[name]
    assert isinstance(array, zarr.Array)
    return np.asarray(array[:])


def _faults(reporter: object) -> list[FaultType]:
    return reporter.kinds()  # ty: ignore[unresolved-attribute]


def test_a_complete_run_is_archived_with_the_providers_member_numbers(tmp_path: Path) -> None:
    bucket = FakeBucket()
    recorder, reporter = build_mogreps_recorder(tmp_path, bucket, Clock(SOON))
    assert recorder.run_cycle([TINY_MOGREPS])
    assert _status(tmp_path) == STATUS_COMPLETE
    assert _faults(reporter) == []
    slot = slot_for(TINY_MOGREPS, INIT)
    assert _stored(tmp_path, "realization")[slot].tolist() == list(REALIZATIONS)
    assert list(np.asarray(_group(tmp_path).attrs["grid_shape"])) == list(GRID.shape or ())
    for member in (1, 2, 3):
        file = _file("wind_100m", 60, member)
        stored = _stored(tmp_path, "wind_100m")[slot, member - 1, 1]
        np.testing.assert_allclose(stored, expected_crop(file), rtol=2**-12)
    # Lead time 0 of the shortwave field does not exist, so its slot on the step axis stays NaN.
    assert np.isnan(_stored(tmp_path, "sw")[slot, 0, 0]).all()
    assert not ProductCache(root=tmp_path / "cache", product=TINY_MOGREPS.name).cached_runs()


def test_a_live_run_is_handled_before_the_backfill_and_the_backfill_has_a_time_budget(
    tmp_path: Path,
) -> None:
    old = hours_after(INIT, -20)
    bucket = FakeBucket()
    clock = Clock(SOON)
    recorder, _ = build_mogreps_recorder(
        tmp_path, bucket, clock, lookback_hours=24.0, backfill_seconds=0.0
    )
    recorder.run_cycle([TINY_MOGREPS])
    assert _status(tmp_path, INIT) == STATUS_COMPLETE
    assert _status(tmp_path, old) == 0
    assert bucket.urls_of_run(old) == []


def test_the_backfill_records_the_older_runs_newest_first(tmp_path: Path) -> None:
    older = [hours_after(INIT, -offset) for offset in (8, 10)]
    bucket = FakeBucket()
    recorder, _ = build_mogreps_recorder(
        tmp_path, bucket, Clock(SOON), lookback_hours=14.0, backfill_seconds=3600.0
    )
    recorder.run_cycle([TINY_MOGREPS])
    assert all(_status(tmp_path, init) == STATUS_COMPLETE for init in [INIT, *older])
    first_urls = [bucket.requests.index(bucket.urls_of_run(init)[0]) for init in older]
    assert first_urls[0] < first_urls[1]


def test_a_run_with_no_files_is_looked_at_again_only_after_a_growing_delay(
    tmp_path: Path,
) -> None:
    bucket = FakeBucket(published=lambda _file: False)
    clock = Clock(SOON)
    recorder, reporter = build_mogreps_recorder(tmp_path, bucket, clock, lookback_hours=3.0)
    recorder.run_cycle([TINY_MOGREPS])
    after_first = len(bucket.urls_of_run(INIT))
    assert after_first > 0
    clock.advance(timedelta(minutes=15))
    recorder.run_cycle([TINY_MOGREPS])
    assert len(bucket.urls_of_run(INIT)) == after_first
    clock.advance(timedelta(minutes=30))
    recorder.run_cycle([TINY_MOGREPS])
    after_third = len(bucket.urls_of_run(INIT))
    assert after_third > after_first
    clock.advance(timedelta(minutes=30))
    recorder.run_cycle([TINY_MOGREPS])
    assert len(bucket.urls_of_run(INIT)) == after_third
    assert _faults(reporter) == []


def test_a_run_absent_for_its_whole_window_is_recorded_missing_once(tmp_path: Path) -> None:
    # The archive needs a grid before it can record anything, so a later run is published.
    published_run = hours_after(INIT, 28 * 24)
    bucket = FakeBucket(published=lambda file: file.init_time == published_run)
    clock = Clock(hours_after(published_run, 2.5))
    recorder, reporter = build_mogreps_recorder(
        tmp_path, bucket, clock, lookback_hours=30 * 24.0, backfill_seconds=3600.0
    )
    recorder.run_cycle([TINY_MOGREPS])
    assert _status(tmp_path, published_run) == STATUS_COMPLETE
    assert _status(tmp_path) == 0

    def faults_of_run() -> list[FaultType]:
        return [fault for _, init, fault, _ in reporter.faults if init == INIT]

    assert faults_of_run() == []
    clock.now = hours_after(INIT, 29 * 24 + 1)
    recorder.run_cycle([TINY_MOGREPS])
    assert _status(tmp_path) == STATUS_MISSING
    recorder.run_cycle([TINY_MOGREPS])
    assert faults_of_run().count("missing") == 1


def test_a_run_with_some_files_waits_for_the_deadline_then_commits_partial(
    tmp_path: Path,
) -> None:
    def published(file: ExpectedFile) -> bool:
        return file.init_time == INIT and not (
            file.field.variable == "sw" and file.step_minutes == 120
        )

    bucket = FakeBucket(published=published)
    clock = Clock(hours_after(INIT, 23))
    recorder, reporter = build_mogreps_recorder(tmp_path, bucket, clock, lookback_hours=30.0)
    recorder.run_cycle([TINY_MOGREPS])
    assert _status(tmp_path) == 0
    clock.now = hours_after(INIT, 24.5)
    recorder.run_cycle([TINY_MOGREPS])
    assert _status(tmp_path) == STATUS_PARTIAL
    assert _faults(reporter) == ["partial"]
    slot = slot_for(TINY_MOGREPS, INIT)
    assert _stored(tmp_path, "files_received")[slot] == 12
    assert _stored(tmp_path, "files_expected")[slot] == 15


def test_a_file_that_lands_late_is_still_fetched_before_the_deadline(tmp_path: Path) -> None:
    late = {"published": False}

    def published(file: ExpectedFile) -> bool:
        return late["published"] or file.step_minutes != 120

    bucket = FakeBucket(published=published)
    clock = Clock(SOON)
    recorder, _ = build_mogreps_recorder(tmp_path, bucket, clock)
    recorder.run_cycle([TINY_MOGREPS])
    assert _status(tmp_path) == 0
    late["published"] = True
    clock.advance(timedelta(minutes=30))
    recorder.run_cycle([TINY_MOGREPS])
    assert _status(tmp_path) == STATUS_COMPLETE


def test_a_budget_that_ends_mid_run_leaves_an_old_run_waiting_and_resumes_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    old = hours_after(INIT, -30)  # older than the 24 h partial deadline
    bucket = FakeBucket(published=lambda file: file.init_time in {old, INIT})
    recorder, reporter = build_mogreps_recorder(
        tmp_path,
        bucket,
        Clock(SOON),
        lookback_hours=36.0,
        backfill_seconds=10.0,
    )
    real = recorder.__class__._fetch_pass
    now = {"seconds": 0.0}

    def expire_after_five_files(self: object, **kwargs: object) -> object:
        # The budget runs out once eight files of the run have been tried.
        calls = {"n": 0}

        def stop() -> bool:
            calls["n"] += 1
            if calls["n"] > 8:
                now["seconds"] = 100.0
            return now["seconds"] > 10.0

        if kwargs["stop"] is not None:
            kwargs["stop"] = stop
        return real(self, **kwargs)  # ty: ignore[invalid-argument-type]

    monkeypatch.setattr(recorder.__class__, "_fetch_pass", expire_after_five_files)
    monkeypatch.setattr(recorder_module.time, "monotonic", lambda: now["seconds"])
    recorder.run_cycle([TINY_MOGREPS])
    assert _status(tmp_path, old) == 0
    assert "partial" not in _faults(reporter)
    assert ProductCache(root=tmp_path / "cache", product=TINY_MOGREPS.name).cached_runs()
    # Next cycle, with no budget pressure, finishes the run as complete.
    monkeypatch.setattr(recorder.__class__, "_fetch_pass", real)
    now["seconds"] = 0.0
    recorder.run_cycle([TINY_MOGREPS])
    assert _status(tmp_path, old) == STATUS_COMPLETE
    assert "partial" not in _faults(reporter)


def test_the_backfill_of_runs_older_than_the_deadline_goes_newest_first(tmp_path: Path) -> None:
    older = [hours_after(INIT, -offset) for offset in (26, 30, 34)]
    bucket = FakeBucket()
    recorder, _ = build_mogreps_recorder(
        tmp_path, bucket, Clock(SOON), lookback_hours=40.0, backfill_seconds=3600.0
    )
    recorder.run_cycle([TINY_MOGREPS])
    firsts = [bucket.requests.index(bucket.urls_of_run(init)[0]) for init in older]
    assert firsts == sorted(firsts)
    assert all(_status(tmp_path, init) == STATUS_COMPLETE for init in older)
