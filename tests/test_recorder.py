import errno
import logging
import re
import shutil
from collections.abc import Callable
from datetime import timedelta
from pathlib import Path

import httpx
import numpy as np
import pytest
import zarr
from support import (
    BASE_URL,
    CLAT,
    INIT,
    N_KEPT_CELLS,
    TINY_DETERMINISTIC,
    TINY_EPS,
    Clock,
    FakeProvider,
    build_recorder,
    data_values,
    make_grib,
)

from nwp_archivist import recorder as recorder_module
from nwp_archivist.cache import ProductCache, RunCache
from nwp_archivist.products import ExpectedFile, Product, expected_files, run_directory
from nwp_archivist.recorder import RecorderConfig
from nwp_archivist.store import (
    STATUS_COMPLETE,
    STATUS_MISSING,
    STATUS_PARTIAL,
    ProductStore,
    RunToCommit,
    StoreLocation,
    round_significand,
    slot_for,
)

# The run is fetched from one hour after initialisation; its deadline is 12 hours after.
SOON = INIT + timedelta(hours=1, minutes=30)
DEADLINE = INIT + timedelta(hours=12)
LOOKBACK_ONE_RUN = 2.0
# A lookback wide enough to reach the run at its deadline, though not earlier ones.
LOOKBACK_TO_DEADLINE = 13.0


def _only_run(
    *, skip: Callable[[ExpectedFile], bool] | None = None
) -> Callable[[ExpectedFile], bool]:
    """Publish only the files of one run, optionally withholding some of them."""

    def published(file: ExpectedFile) -> bool:
        return run_directory(INIT) in file.url and not (skip and skip(file))

    return published


def _store(tmp_path: Path, product: Product = TINY_EPS) -> ProductStore:
    return ProductStore.open(location=StoreLocation(root=str(tmp_path / "store")), product=product)


def _status(tmp_path: Path, product: Product = TINY_EPS) -> int:
    statuses = _store(tmp_path, product).statuses()
    return int(statuses[slot_for(product, INIT)]) if len(statuses) > slot_for(product, INIT) else 0


def _run_files(product: Product = TINY_EPS) -> list[ExpectedFile]:
    return expected_files(product, INIT, base_url=BASE_URL)


def test_a_complete_run_is_committed_complete_and_its_local_files_are_deleted(
    tmp_path: Path,
) -> None:
    provider = FakeProvider(TINY_EPS, published=_only_run())
    recorder, reporter = build_recorder(
        tmp_path, provider, Clock(SOON), lookback_hours=LOOKBACK_ONE_RUN
    )
    assert recorder.run_cycle([TINY_EPS])
    assert _status(tmp_path) == STATUS_COMPLETE
    assert reporter.faults == []
    assert reporter.check_ins == [True]
    cache = ProductCache(root=tmp_path / "cache", product=TINY_EPS.name)
    assert not cache.run(INIT).directory.exists()


def test_a_run_after_complete_fetches_nothing(tmp_path: Path) -> None:
    provider = FakeProvider(TINY_EPS, published=_only_run())
    recorder, _ = build_recorder(tmp_path, provider, Clock(SOON), lookback_hours=LOOKBACK_ONE_RUN)
    recorder.run_cycle([TINY_EPS])
    before = len(provider.requests)
    recorder.run_cycle([TINY_EPS])
    assert len(provider.requests) == before


def test_a_rebuilt_instance_never_commits_a_run_twice(tmp_path: Path) -> None:
    provider = FakeProvider(TINY_EPS, published=_only_run())
    recorder, _ = build_recorder(tmp_path, provider, Clock(SOON), lookback_hours=LOOKBACK_ONE_RUN)
    recorder.run_cycle([TINY_EPS])
    shutil.rmtree(tmp_path / "cache")
    before = len(provider.requests)
    recorder.run_cycle([TINY_EPS])
    assert len(provider.requests) == before


def test_a_deterministic_product_commits_complete(tmp_path: Path) -> None:
    provider = FakeProvider(TINY_DETERMINISTIC, published=_only_run())
    recorder, reporter = build_recorder(
        tmp_path, provider, Clock(SOON), lookback_hours=LOOKBACK_ONE_RUN
    )
    recorder.run_cycle([TINY_DETERMINISTIC])
    assert _status(tmp_path, TINY_DETERMINISTIC) == STATUS_COMPLETE
    assert reporter.faults == []


def test_before_the_deadline_a_missing_file_leaves_the_run_waiting(tmp_path: Path) -> None:
    last = _run_files()[-1]
    provider = FakeProvider(TINY_EPS, published=_only_run(skip=lambda f: f == last))
    recorder, reporter = build_recorder(
        tmp_path, provider, Clock(SOON), lookback_hours=LOOKBACK_ONE_RUN
    )
    assert recorder.run_cycle([TINY_EPS])
    assert _status(tmp_path) == 0
    assert reporter.faults == []


def test_at_the_deadline_a_run_with_all_but_one_file_is_committed_partial(
    tmp_path: Path,
) -> None:
    files = _run_files()
    last = files[-1]
    provider = FakeProvider(TINY_EPS, published=_only_run(skip=lambda f: f == last))
    clock = Clock(DEADLINE)
    recorder, reporter = build_recorder(
        tmp_path, provider, clock, lookback_hours=LOOKBACK_TO_DEADLINE
    )
    assert recorder.run_cycle([TINY_EPS])
    assert _status(tmp_path) == STATUS_PARTIAL
    store = _store(tmp_path)
    slot = slot_for(TINY_EPS, INIT)
    session = store.repository.readonly_session(branch="main")
    group = zarr.open_group(session.store, mode="r")
    expected, received = group["files_expected"], group["files_received"]
    assert isinstance(expected, zarr.Array)
    assert isinstance(received, zarr.Array)
    assert expected[slot] == len(files)
    assert received[slot] == len(files) - 1
    assert reporter.kinds() == ["partial"]
    assert reporter.faults[0][0] == TINY_EPS.name
    assert reporter.faults[0][1] == INIT


def test_at_the_deadline_a_run_with_no_files_is_committed_missing_and_reported_once(
    tmp_path: Path,
) -> None:
    provider = FakeProvider(TINY_EPS, published=lambda _file: False)
    recorder, reporter = build_recorder(
        tmp_path, provider, Clock(DEADLINE), lookback_hours=LOOKBACK_TO_DEADLINE
    )
    recorder.run_cycle([TINY_EPS])
    recorder.run_cycle([TINY_EPS])
    assert _status(tmp_path) == STATUS_MISSING
    missing_for_run = [f for f in reporter.faults if f[1] == INIT and f[2] == "missing"]
    assert len(missing_for_run) == 1


def test_a_first_run_whose_grid_never_arrives_is_reported_missing_once(tmp_path: Path) -> None:
    provider = FakeProvider(TINY_EPS, published=lambda _file: False, static_available=False)
    recorder, reporter = build_recorder(
        tmp_path, provider, Clock(DEADLINE), lookback_hours=LOOKBACK_TO_DEADLINE
    )
    recorder.run_cycle([TINY_EPS])
    recorder.run_cycle([TINY_EPS])
    assert [f[2] for f in reporter.faults if f[1] == INIT] == ["missing"]


def test_a_restart_before_the_deadline_with_a_half_filled_cache_fetches_only_the_rest(
    tmp_path: Path,
) -> None:
    def first_member_only(file: ExpectedFile) -> bool:
        return file.member == 1

    provider = FakeProvider(TINY_EPS, published=_only_run(skip=lambda f: f.member == 2))
    clock = Clock(SOON)
    recorder, reporter = build_recorder(tmp_path, provider, clock, lookback_hours=LOOKBACK_ONE_RUN)
    recorder.run_cycle([TINY_EPS])
    assert _status(tmp_path) == 0
    member_one_urls = {f.url for f in _run_files() if first_member_only(f)}
    provider.published = _only_run()
    provider.requests.clear()
    clock.now = DEADLINE - timedelta(hours=1)
    recorder, reporter = build_recorder(
        tmp_path, provider, clock, reporter=reporter, lookback_hours=13.0
    )
    recorder.run_cycle([TINY_EPS])
    assert _status(tmp_path) == STATUS_COMPLETE
    assert not member_one_urls & set(provider.requests)
    assert reporter.faults == []


def test_a_restart_long_after_the_deadline_reports_the_run_missing_exactly_once(
    tmp_path: Path,
) -> None:
    provider = FakeProvider(TINY_EPS, published=lambda _file: False)
    clock = Clock(INIT + timedelta(hours=30))
    recorder, reporter = build_recorder(tmp_path, provider, clock, lookback_hours=40.0)
    recorder.run_cycle([TINY_EPS])
    clock.advance(timedelta(minutes=15))
    recorder.run_cycle([TINY_EPS])
    assert [f[2] for f in reporter.faults if f[1] == INIT] == ["missing"]


def test_a_truncated_response_leaves_the_file_not_received(tmp_path: Path) -> None:
    target = _run_files()[0]
    body = make_grib(data_values(target), short_name="2t", step_minutes=0, member=1)
    provider = FakeProvider(TINY_EPS, published=_only_run())
    provider.overrides[target.url] = httpx.Response(
        200, content=body[:-20], headers={"content-length": str(len(body))}
    )
    recorder, _ = build_recorder(tmp_path, provider, Clock(SOON), lookback_hours=LOOKBACK_ONE_RUN)
    recorder.run_cycle([TINY_EPS])
    assert _status(tmp_path) == 0
    run_cache = ProductCache(root=tmp_path / "cache", product=TINY_EPS.name).run(INIT)
    assert not run_cache.has(target, n_cells=N_KEPT_CELLS)
    # The rest of the sequence waits for the next cycle, so the file is retried first.
    assert run_cache.count_received(_run_files(), n_cells=N_KEPT_CELLS) == len(_run_files()) - 3


def test_a_header_that_disagrees_with_the_url_leaves_the_file_not_received(
    tmp_path: Path,
) -> None:
    target = _run_files()[0]
    wrong = make_grib(data_values(target), short_name="2t", step_minutes=60, member=1)
    provider = FakeProvider(TINY_EPS, published=_only_run())
    provider.overrides[target.url] = httpx.Response(200, content=wrong)
    recorder, _ = build_recorder(tmp_path, provider, Clock(SOON), lookback_hours=LOOKBACK_ONE_RUN)
    recorder.run_cycle([TINY_EPS])
    run_cache = ProductCache(root=tmp_path / "cache", product=TINY_EPS.name).run(INIT)
    assert not run_cache.has(target, n_cells=N_KEPT_CELLS)
    assert _status(tmp_path) == 0


def test_a_full_disk_in_one_run_reports_once_and_lets_the_next_run_proceed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_save = RunCache.save

    def save(self: RunCache, file: ExpectedFile, values: np.ndarray) -> None:
        if self.directory.name == INIT.strftime("%Y%m%dT%H%M"):
            raise OSError(errno.ENOSPC, "No space left on device")
        real_save(self, file, values)

    monkeypatch.setattr(RunCache, "save", save)
    provider = FakeProvider(TINY_EPS)
    clock = Clock(INIT + timedelta(hours=7, minutes=30))
    recorder, reporter = build_recorder(tmp_path, provider, clock, lookback_hours=8.0)
    assert not recorder.run_cycle([TINY_EPS])
    assert [f[2] for f in reporter.faults] == ["run_error"]
    assert reporter.faults[0][1] == INIT
    assert reporter.check_ins == [False]
    later = INIT + timedelta(hours=6)
    assert _store(tmp_path).is_archived(later)
    assert not _store(tmp_path).is_archived(INIT)


def test_a_storage_error_at_commit_reports_once_and_leaves_the_run_expected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def failing_commit(self: ProductStore, run: RunToCommit) -> str:
        message = "S3 is down"
        raise OSError(message)

    provider = FakeProvider(TINY_EPS, published=_only_run())
    recorder, reporter = build_recorder(
        tmp_path, provider, Clock(SOON), lookback_hours=LOOKBACK_ONE_RUN
    )
    with monkeypatch.context() as patch:
        patch.setattr(ProductStore, "commit_run", failing_commit)
        assert not recorder.run_cycle([TINY_EPS])
    assert reporter.kinds() == ["commit_failed"]
    cache = ProductCache(root=tmp_path / "cache", product=TINY_EPS.name)
    assert cache.run(INIT).directory.exists()
    fetched = len(provider.requests)
    assert recorder.run_cycle([TINY_EPS])
    assert _status(tmp_path) == STATUS_COMPLETE
    # The second cycle refetches only the grid check, not the cached data files.
    assert len(provider.requests) - fetched <= 2


def test_an_unopenable_repository_reports_and_does_not_raise(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def failing_open(*, location: StoreLocation, product: Product) -> ProductStore:
        message = "no route to host"
        raise OSError(message)

    monkeypatch.setattr(ProductStore, "open", failing_open)
    recorder, reporter = build_recorder(tmp_path, FakeProvider(TINY_EPS), Clock(SOON))
    assert not recorder.run_cycle([TINY_EPS])
    assert reporter.kinds() == ["commit_failed"]


def test_a_changed_grid_stops_commits_and_reports_once(tmp_path: Path) -> None:
    provider = FakeProvider(TINY_EPS)
    clock = Clock(INIT + timedelta(hours=1, minutes=30))
    recorder, reporter = build_recorder(tmp_path, provider, clock, lookback_hours=2.0)
    recorder.run_cycle([TINY_EPS])
    assert _status(tmp_path) == STATUS_COMPLETE
    moved = CLAT.copy()
    moved[1] += 0.25
    provider.clat = moved
    clock.now = INIT + timedelta(hours=7, minutes=30)
    recorder, _ = build_recorder(tmp_path, provider, clock, reporter=reporter, lookback_hours=2.0)
    recorder.run_cycle([TINY_EPS])
    recorder.run_cycle([TINY_EPS])
    assert reporter.kinds() == ["grid_changed"]
    assert not _store(tmp_path).is_archived(INIT + timedelta(hours=6))
    assert ProductCache(root=tmp_path / "cache", product=TINY_EPS.name).halted() is not None


def test_a_low_disk_reports_a_fault_naming_the_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        recorder_module.shutil,
        "disk_usage",
        lambda _path: shutil._ntuple_diskusage(100, 99, 1),
    )
    provider = FakeProvider(TINY_EPS, published=_only_run())
    recorder, reporter = build_recorder(
        tmp_path, provider, Clock(SOON), lookback_hours=LOOKBACK_ONE_RUN
    )
    recorder.run_cycle([TINY_EPS])
    assert reporter.kinds() == ["disk_low"]
    assert reporter.faults[0][1] is None
    assert _status(tmp_path) == STATUS_COMPLETE


def test_before_the_deadline_each_sequence_stops_at_its_first_missing_file(
    tmp_path: Path,
) -> None:
    provider = FakeProvider(TINY_EPS, published=lambda _file: False)
    recorder, _ = build_recorder(tmp_path, provider, Clock(SOON), lookback_hours=LOOKBACK_ONE_RUN)
    recorder.run_cycle([TINY_EPS])
    # Four grid files, plus one request for each of the four variable-and-member sequences.
    assert len(provider.requests) == 4 + 4


def test_after_the_deadline_every_file_is_tried(tmp_path: Path) -> None:
    provider = FakeProvider(TINY_EPS, published=lambda _file: False)
    recorder, _ = build_recorder(
        tmp_path, provider, Clock(DEADLINE), lookback_hours=LOOKBACK_TO_DEADLINE
    )
    recorder.run_cycle([TINY_EPS])
    data_requests = [u for u in provider.requests if run_directory(INIT) in u and "/p/T_2M/" in u]
    assert len(data_requests) == 2 * 3


def test_each_examined_run_logs_one_line_per_cycle(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    provider = FakeProvider(TINY_EPS, published=_only_run())
    recorder, _ = build_recorder(tmp_path, provider, Clock(SOON), lookback_hours=LOOKBACK_ONE_RUN)
    with caplog.at_level(logging.INFO, logger="nwp_archivist.recorder"):
        recorder.run_cycle([TINY_EPS])
    lines = [r.getMessage() for r in caplog.records if "state=" in r.getMessage()]
    expected = len(_run_files())
    assert lines == [
        (
            f"product=tiny-eps init_time={INIT.isoformat()} expected={expected} "
            f"received={expected} state=complete"
        )
    ]


def _cache(tmp_path: Path) -> ProductCache:
    return ProductCache(root=tmp_path / "cache", product=TINY_EPS.name)


def _stored(tmp_path: Path, variable: str) -> np.ndarray:
    session = _store(tmp_path).repository.readonly_session(branch="main")
    array = zarr.open_group(session.store, mode="r")[variable]
    assert isinstance(array, zarr.Array)
    return np.asarray(array[...])


def test_stored_values_equal_the_served_values_cropped_and_rounded(tmp_path: Path) -> None:
    provider = FakeProvider(TINY_EPS, published=_only_run())
    recorder, _ = build_recorder(tmp_path, provider, Clock(SOON), lookback_hours=LOOKBACK_ONE_RUN)
    recorder.run_cycle([TINY_EPS])
    slot = slot_for(TINY_EPS, INIT)
    cell_index = np.array([0, 1, 2, 3])
    for file in _run_files():
        stored = _stored(tmp_path, file.field.variable)
        assert file.member is not None
        step = TINY_EPS.step_index(file.step_minutes)
        expected = round_significand(data_values(file)[cell_index])
        np.testing.assert_array_equal(stored[slot, file.member - 1, step], expected)


def test_the_default_lookback_is_27_hours() -> None:
    assert RecorderConfig(store=StoreLocation(root="x"), cache_dir=Path("y")).lookback_hours == 27.0


def test_a_cached_run_outside_the_lookback_is_still_committed_and_reported(
    tmp_path: Path,
) -> None:
    provider = FakeProvider(TINY_EPS, published=_only_run(skip=lambda f: f.member == 2))
    clock = Clock(SOON)
    recorder, _ = build_recorder(tmp_path, provider, clock, lookback_hours=LOOKBACK_ONE_RUN)
    recorder.run_cycle([TINY_EPS])
    clock.now = INIT + timedelta(hours=40)
    recorder, reporter = build_recorder(tmp_path, provider, clock)
    recorder.run_cycle([TINY_EPS])
    assert _status(tmp_path) == STATUS_PARTIAL
    assert [f[2] for f in reporter.faults if f[1] == INIT] == ["partial"]
    assert not _cache(tmp_path).run(INIT).directory.exists()


def test_a_truncated_cache_file_is_fetched_again_and_the_run_completes(tmp_path: Path) -> None:
    provider = FakeProvider(TINY_EPS, published=_only_run(skip=lambda f: f.member == 2))
    clock = Clock(SOON)
    recorder, _ = build_recorder(tmp_path, provider, clock, lookback_hours=LOOKBACK_ONE_RUN)
    recorder.run_cycle([TINY_EPS])
    run_directory_ = _cache(tmp_path).run(INIT).directory
    damaged = next((run_directory_ / "T_2M").glob("01_*.npy"))
    damaged.write_bytes(damaged.read_bytes()[:20])
    (run_directory_ / "grid.npz").write_bytes(b"not a zip")
    provider.published = _only_run()
    clock.advance(timedelta(minutes=15))
    recorder, reporter = build_recorder(tmp_path, provider, clock, lookback_hours=LOOKBACK_ONE_RUN)
    assert recorder.run_cycle([TINY_EPS])
    assert _status(tmp_path) == STATUS_COMPLETE
    assert reporter.faults == []
    np.testing.assert_array_equal(
        _stored(tmp_path, "T_2M")[slot_for(TINY_EPS, INIT), 0, 0],
        round_significand(data_values(_run_files()[0])[:N_KEPT_CELLS]),
    )


def test_a_halted_product_keeps_fetching_but_does_not_commit(tmp_path: Path) -> None:
    provider = FakeProvider(TINY_EPS)
    clock = Clock(SOON)
    recorder, reporter = build_recorder(tmp_path, provider, clock, lookback_hours=LOOKBACK_ONE_RUN)
    recorder.run_cycle([TINY_EPS])
    moved = CLAT.copy()
    moved[1] += 0.25
    provider.clat = moved
    clock.now = INIT + timedelta(hours=7, minutes=30)
    recorder, _ = build_recorder(
        tmp_path, provider, clock, reporter=reporter, lookback_hours=LOOKBACK_ONE_RUN
    )
    recorder.run_cycle([TINY_EPS])
    later = INIT + timedelta(hours=6)
    assert not _store(tmp_path).is_archived(later)
    cached = _cache(tmp_path).run(later)
    assert cached.count_received(
        expected_files(TINY_EPS, later, base_url=BASE_URL), n_cells=N_KEPT_CELLS
    ) == len(_run_files())
    assert reporter.kinds() == ["grid_changed"]


def test_a_data_file_with_a_different_cell_count_halts_the_product(tmp_path: Path) -> None:
    target = _run_files()[0]
    odd = make_grib(np.arange(7, dtype=np.float32) + 280, short_name="2t", member=1)
    provider = FakeProvider(TINY_EPS, published=_only_run())
    provider.overrides[target.url] = httpx.Response(200, content=odd)
    recorder, reporter = build_recorder(
        tmp_path, provider, Clock(SOON), lookback_hours=LOOKBACK_ONE_RUN
    )
    recorder.run_cycle([TINY_EPS])
    recorder.run_cycle([TINY_EPS])
    assert reporter.kinds() == ["grid_changed"]
    assert "7 cells" in reporter.faults[0][3]
    assert _cache(tmp_path).halted() is not None
    assert _status(tmp_path) == 0


def test_after_the_run_should_be_published_a_skipped_file_does_not_hide_its_sequence(
    tmp_path: Path,
) -> None:
    hole = next(f for f in _run_files() if f.member == 1 and f.field.variable == "T_2M")
    provider = FakeProvider(TINY_EPS, published=_only_run(skip=lambda f: f == hole))
    clock = Clock(INIT + timedelta(hours=5))
    recorder, _ = build_recorder(tmp_path, provider, clock, lookback_hours=LOOKBACK_ONE_RUN + 4)
    recorder.run_cycle([TINY_EPS])
    received = _cache(tmp_path).run(INIT).count_received(_run_files(), n_cells=N_KEPT_CELLS)
    assert received == len(_run_files()) - 1


def test_a_network_blip_at_the_deadline_does_not_commit_the_run_partial(tmp_path: Path) -> None:
    target = _run_files()[3]
    provider = FakeProvider(TINY_EPS, published=_only_run())
    # The fetcher makes 3 attempts per pass, so 4 failures outlast the first pass.
    provider.failures[target.url] = 4
    recorder, reporter = build_recorder(
        tmp_path, provider, Clock(DEADLINE), lookback_hours=LOOKBACK_TO_DEADLINE
    )
    recorder.run_cycle([TINY_EPS])
    assert _status(tmp_path) == STATUS_COMPLETE
    assert "partial" not in reporter.kinds()


def test_a_persistent_commit_failure_is_reported_once_until_it_clears(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def failing_commit(self: ProductStore, run: RunToCommit) -> str:
        message = "S3 is down"
        raise OSError(message)

    provider = FakeProvider(TINY_EPS, published=_only_run())
    clock = Clock(SOON)
    recorder, reporter = build_recorder(tmp_path, provider, clock, lookback_hours=LOOKBACK_ONE_RUN)
    with monkeypatch.context() as patch:
        patch.setattr(ProductStore, "commit_run", failing_commit)
        for _ in range(3):
            recorder.run_cycle([TINY_EPS])
            clock.advance(timedelta(minutes=15))
    assert reporter.kinds() == ["commit_failed"]
    assert _cache(tmp_path).load_active_faults() == {f"commit_failed:{INIT.isoformat()}"}
    recorder.run_cycle([TINY_EPS])
    assert _status(tmp_path) == STATUS_COMPLETE
    assert _cache(tmp_path).load_active_faults() == set()


def test_a_persistent_low_disk_is_reported_once_per_product_not_per_cycle_or_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        recorder_module.shutil, "disk_usage", lambda _path: shutil._ntuple_diskusage(100, 99, 1)
    )
    provider = FakeProvider(TINY_EPS, published=lambda _file: False)
    clock = Clock(SOON)
    recorder, reporter = build_recorder(
        tmp_path, provider, clock, lookback_hours=LOOKBACK_TO_DEADLINE
    )
    recorder.run_cycle([TINY_EPS])
    clock.advance(timedelta(minutes=15))
    recorder.run_cycle([TINY_EPS])
    assert reporter.kinds() == ["disk_low"]


def test_a_leftover_cache_of_an_archived_run_is_deleted(tmp_path: Path) -> None:
    provider = FakeProvider(TINY_EPS, published=_only_run())
    recorder, _ = build_recorder(tmp_path, provider, Clock(SOON), lookback_hours=LOOKBACK_ONE_RUN)
    recorder.run_cycle([TINY_EPS])
    leftover = _cache(tmp_path).run(INIT)
    leftover.save_generating_process(11)
    assert leftover.directory.exists()
    recorder.run_cycle([TINY_EPS])
    assert not leftover.directory.exists()


def test_each_cycle_logs_the_age_of_the_oldest_run_it_fetched_from(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    provider = FakeProvider(TINY_EPS, published=_only_run())
    recorder, _ = build_recorder(tmp_path, provider, Clock(SOON), lookback_hours=LOOKBACK_ONE_RUN)
    with caplog.at_level(logging.INFO, logger="nwp_archivist.recorder"):
        recorder.run_cycle([TINY_EPS])
    assert "product=tiny-eps oldest_fetched_age_hours=1.5" in [
        r.getMessage() for r in caplog.records
    ]


def test_the_code_version_carries_the_git_commit_of_the_checkout() -> None:
    assert re.fullmatch(r".+\+[0-9a-f]{12}", recorder_module.code_version())


def test_the_code_version_falls_back_to_the_package_version_without_git(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def no_git(*_args: object, **_kwargs: object) -> None:
        raise FileNotFoundError

    monkeypatch.setattr(recorder_module.subprocess, "run", no_git)
    assert re.fullmatch(r"\d+\.\d+\.\d+", recorder_module.code_version())
