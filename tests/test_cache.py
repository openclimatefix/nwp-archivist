from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from support import BASE_URL, INIT, N_KEPT_CELLS, TINY_EPS

from nwp_archivist.cache import ProductCache, RunCache, RunGrid
from nwp_archivist.products import ExpectedFile, expected_files


def _file() -> ExpectedFile:
    return expected_files(TINY_EPS, INIT, base_url=BASE_URL)[0]


def _run(tmp_path: Path) -> RunCache:
    return ProductCache(root=tmp_path, product="tiny-eps").run(INIT)


def test_a_whole_file_is_received(tmp_path: Path) -> None:
    run = _run(tmp_path)
    run.save(_file(), np.zeros(N_KEPT_CELLS, dtype=np.float32))
    assert run.has(_file(), n_cells=N_KEPT_CELLS)


def test_a_file_of_the_wrong_size_is_not_received_and_is_deleted(tmp_path: Path) -> None:
    run = _run(tmp_path)
    run.save(_file(), np.zeros(N_KEPT_CELLS, dtype=np.float32))
    path = next(run.directory.rglob("*.npy"))
    path.write_bytes(b"")
    assert not run.has(_file(), n_cells=N_KEPT_CELLS)
    assert not path.exists()


def test_an_unreadable_file_loads_as_none_and_is_deleted(tmp_path: Path) -> None:
    run = _run(tmp_path)
    run.save(_file(), np.zeros(N_KEPT_CELLS, dtype=np.float32))
    path = next(run.directory.rglob("*.npy"))
    path.write_bytes(b"garbage")
    assert run.load(variable="T_2M", member=1, step_minutes=0) is None
    assert not path.exists()


def test_an_unreadable_grid_loads_as_none_and_is_deleted(tmp_path: Path) -> None:
    run = _run(tmp_path)
    grid = RunGrid(n_points=6, cell_index=np.arange(4, dtype=np.int32), statics={})
    run.save_grid(grid)
    (run.directory / "grid.npz").write_bytes(b"garbage")
    assert run.load_grid() is None
    assert not (run.directory / "grid.npz").exists()


def test_an_unreadable_metadata_file_loads_as_none(tmp_path: Path) -> None:
    run = _run(tmp_path)
    run.save_generating_process(11)
    (run.directory / "meta.json").write_text("{")
    assert run.load_generating_process() is None


def test_cached_runs_lists_the_init_times_oldest_first(tmp_path: Path) -> None:
    cache = ProductCache(root=tmp_path, product="tiny-eps")
    later = datetime(2026, 9, 25, 12, tzinfo=UTC)
    cache.run(later).save_generating_process(1)
    cache.run(INIT).save_generating_process(1)
    (cache.directory / "runs" / "not-a-time").mkdir()
    assert cache.cached_runs() == [INIT, later]


def test_active_faults_round_trip_and_a_missing_file_means_none(tmp_path: Path) -> None:
    cache = ProductCache(root=tmp_path, product="tiny-eps")
    assert cache.load_active_faults() == set()
    cache.save_active_faults({"a", "b"})
    assert cache.load_active_faults() == {"a", "b"}


def test_a_grid_keeps_its_rectangle_shape_through_the_cache(tmp_path: Path) -> None:
    run = _run(tmp_path)
    grid = RunGrid(
        n_points=100,
        cell_index=np.arange(6, dtype=np.int32),
        statics={"clat": np.zeros(6, np.float32)},
        shape=(2, 3),
    )
    run.save_grid(grid)
    loaded = run.load_grid()
    assert loaded is not None
    assert loaded.shape == (2, 3)
    run.save_grid(RunGrid(n_points=100, cell_index=grid.cell_index, statics=grid.statics))
    loaded = run.load_grid()
    assert loaded is not None
    assert loaded.shape is None


def test_member_ids_round_trip_and_an_unreadable_file_is_dropped(tmp_path: Path) -> None:
    run = _run(tmp_path)
    assert run.load_member_ids() is None
    run.save_member_ids((3, 4, 5))
    assert run.load_member_ids() == (3, 4, 5)
    (run.directory / "members.json").write_text("{")
    assert run.load_member_ids() is None
    assert not (run.directory / "members.json").exists()


def test_a_backoff_round_trips_and_an_unreadable_file_is_dropped(tmp_path: Path) -> None:
    run = _run(tmp_path)
    assert run.load_backoff() is None
    when = datetime(2026, 9, 25, 12, 30, tzinfo=UTC)
    run.save_backoff(attempts=2, next_try=when)
    assert run.load_backoff() == (2, when)
    (run.directory / "backoff.json").write_text("[]")
    assert run.load_backoff() is None
