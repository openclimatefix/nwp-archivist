from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import boto3
import numpy as np
import pytest
import xarray as xr
import zarr
from support import CLAT, CLON, INIT, N_KEPT_CELLS, TINY_DETERMINISTIC, TINY_EPS

from nwp_archivist.cache import RunGrid
from nwp_archivist.products import Field, Product
from nwp_archivist.store import (
    STATUS_COMPLETE,
    STATUS_PARTIAL,
    ProductStore,
    RunToCommit,
    StoreLocation,
    box_cell_index,
    round_significand,
    slot_for,
)

Loader = Callable[[str, int | None, int], np.ndarray | None]


def _grid(clat: np.ndarray = CLAT, clon: np.ndarray = CLON) -> RunGrid:
    index = box_cell_index(clat=clat, clon=clon)
    statics = {
        "clat": clat[index],
        "clon": clon[index],
        "hsurf": np.full(len(index), 10.0, dtype=np.float32),
        "fr_land": np.ones(len(index), dtype=np.float32),
    }
    return RunGrid(n_points=len(clat), cell_index=index, statics=statics)


def _source(product: Product, seed: float) -> Loader:
    """Cropped values for every file, as the cache would hold them."""

    def load(variable: str, member: int | None, step: int) -> np.ndarray | None:
        field = product.field_by_variable(variable)
        if step not in field.steps_minutes:
            return None
        base = seed + (member or 0) * 100 + step / 60 * 10 + len(variable)
        return (np.arange(N_KEPT_CELLS, dtype=np.float32) + np.float32(base)).astype(np.float32)

    return load


def _run(
    product: Product,
    init: datetime,
    *,
    seed: float = 280.0,
    status: int = STATUS_COMPLETE,
    load: Loader | None = None,
) -> RunToCommit:
    return RunToCommit(
        init_time=init,
        status=status,
        files_expected=10,
        files_received=10,
        generating_process=7,
        archived_at=datetime(2026, 9, 25, 20, tzinfo=UTC),
        code_version="test",
        load=load or _source(product, seed),
    )


@pytest.fixture
def store(tmp_path: Path) -> ProductStore:
    return ProductStore.open(location=StoreLocation(root=str(tmp_path)), product=TINY_EPS)


def _read(store: ProductStore, name: str) -> np.ndarray:
    """Read a whole array of the archive's latest snapshot."""
    session = store.repository.readonly_session(branch="main")
    array = zarr.open_group(session.store, mode="r")[name]
    assert isinstance(array, zarr.Array)
    return np.asarray(array[...])


def test_box_keeps_exactly_the_cells_inside_and_drops_one_0_01_degrees_outside() -> None:
    index = box_cell_index(clat=CLAT, clon=CLON)
    # Cell 2 sits on the box edge (kept), cell 4 is 0.01 degrees north of it (dropped).
    assert index.tolist() == [0, 1, 2, 3]
    assert index.dtype == np.int32


def test_round_significand_zeroes_the_low_bits_and_bounds_the_error() -> None:
    rng = np.random.default_rng(0)
    values = rng.uniform(-500, 500, 10_000).astype(np.float32)
    rounded = round_significand(values)
    assert not (rounded.view(np.uint32) & 0x7FF).any()
    assert np.all(np.abs(rounded - values) <= 2.0**-13 * np.abs(values))


def test_round_significand_matches_round_to_nearest_on_the_bits() -> None:
    rng = np.random.default_rng(1)
    values = rng.uniform(1, 1000, 10_000).astype(np.float32)
    bits = values.view(np.uint32).astype(np.uint64)
    half_even = ((bits >> 11) & 1) + 0x3FF
    expected = (((bits + half_even) >> 11) << 11).astype(np.uint32).view(np.float32)
    np.testing.assert_array_equal(round_significand(values), expected)


def test_round_significand_keeps_nan_and_is_idempotent() -> None:
    values = np.array([np.nan, 1.2345678, 273.15], dtype=np.float32)
    once = round_significand(values)
    assert np.isnan(once[0])
    np.testing.assert_array_equal(round_significand(once)[1:], once[1:])


def test_round_significand_rejects_float64() -> None:
    with pytest.raises(TypeError):
        round_significand(np.zeros(3))


def test_slot_for_counts_whole_cycles_from_the_epoch() -> None:
    assert slot_for(TINY_EPS, datetime(2026, 1, 1, tzinfo=UTC)) == 0
    assert slot_for(TINY_EPS, datetime(2026, 1, 2, 6, tzinfo=UTC)) == 5
    with pytest.raises(ValueError, match="cycle"):
        slot_for(TINY_EPS, datetime(2026, 1, 1, 3, tzinfo=UTC))


def test_a_committed_run_reads_back_rounded_with_padding(store: ProductStore) -> None:
    store.initialise(_grid())
    store.commit_run(_run(TINY_EPS, INIT))
    slot = slot_for(TINY_EPS, INIT)
    expected = round_significand(np.asarray(_source(TINY_EPS, 280.0)("T_2M", 2, 60)))
    np.testing.assert_array_equal(_read(store, "T_2M")[slot, 1, 1, :], expected)
    # U_10M has two steps, so the third step is NaN padding.
    assert np.isnan(_read(store, "U_10M")[slot, 0, 2, :]).all()
    assert _read(store, "step_of_U_10M").tolist() == [0, 60, -1]
    assert _read(store, "step_of_T_2M").tolist() == [0, 60, 120]


def test_status_arrays_record_the_run(store: ProductStore) -> None:
    store.initialise(_grid())
    store.commit_run(_run(TINY_EPS, INIT, status=STATUS_PARTIAL))
    slot = slot_for(TINY_EPS, INIT)
    assert _read(store, "status")[slot] == STATUS_PARTIAL
    assert _read(store, "files_expected")[slot] == 10
    assert _read(store, "generating_process")[slot] == 7
    assert _read(store, "code_version")[slot] == "test"
    assert store.is_archived(INIT)
    assert not store.is_archived(INIT + timedelta(hours=6))


def test_slots_before_the_first_run_are_not_archived_nans(store: ProductStore) -> None:
    store.initialise(_grid())
    store.commit_run(_run(TINY_EPS, INIT))
    slot = slot_for(TINY_EPS, INIT)
    assert _read(store, "status")[:slot].tolist() == [0] * slot
    assert np.isnan(_read(store, "T_2M")[0]).all()


def test_the_grid_is_stored_once_per_product(store: ProductStore) -> None:
    store.initialise(_grid())
    stored = store.stored_grid()
    assert stored is not None
    assert stored.cell_index.tolist() == [0, 1, 2, 3]
    assert stored.n_points == len(CLAT)
    assert _read(store, "hsurf").shape == (N_KEPT_CELLS,)


def test_writing_the_same_run_twice_gives_identical_arrays_and_one_slot(
    store: ProductStore,
) -> None:
    store.initialise(_grid())
    store.commit_run(_run(TINY_EPS, INIT))
    first = _read(store, "T_2M")
    store.commit_run(_run(TINY_EPS, INIT))
    np.testing.assert_array_equal(first, _read(store, "T_2M"))
    assert (_read(store, "status") != 0).sum() == 1


def test_runs_committed_out_of_order_leave_the_axis_sorted(store: ProductStore) -> None:
    store.initialise(_grid())
    later = INIT + timedelta(hours=12)
    store.commit_run(_run(TINY_EPS, later, seed=300.0))
    store.commit_run(_run(TINY_EPS, INIT, seed=280.0))
    assert np.all(np.diff(_read(store, "init_time")) == 6 * 3600)
    slot_first, slot_later = slot_for(TINY_EPS, INIT), slot_for(TINY_EPS, later)
    temperature = _read(store, "T_2M")
    for slot, seed in ((slot_first, 280.0), (slot_later, 300.0)):
        loaded = np.asarray(_source(TINY_EPS, seed)("T_2M", 1, 0))
        np.testing.assert_array_equal(temperature[slot, 0, 0], round_significand(loaded))
    assert _read(store, "status")[slot_first + 1] == 0


def test_a_run_built_member_by_member_equals_the_same_run_built_whole(
    store: ProductStore,
) -> None:
    store.initialise(_grid())
    store.commit_run(_run(TINY_EPS, INIT))
    load = _source(TINY_EPS, 280.0)
    whole = np.full((2, 3, N_KEPT_CELLS), np.nan, dtype=np.float32)
    for member in (1, 2):
        for index, step in enumerate((0, 60, 120)):
            whole[member - 1, index] = load("T_2M", member, step)
    slot = slot_for(TINY_EPS, INIT)
    np.testing.assert_array_equal(_read(store, "T_2M")[slot], round_significand(whole))


def test_a_deterministic_product_has_no_member_dimension(tmp_path: Path) -> None:
    store = ProductStore.open(
        location=StoreLocation(root=str(tmp_path)), product=TINY_DETERMINISTIC
    )
    store.initialise(_grid())
    store.commit_run(_run(TINY_DETERMINISTIC, INIT))
    session = store.repository.readonly_session(branch="main")
    dataset = xr.open_zarr(session.store, consolidated=False)
    assert dataset["T_2M"].dims == ("init_time", "step", "cell")
    assert "member" not in dataset


def test_an_interrupted_commit_leaves_the_previous_snapshot_readable(
    store: ProductStore, tmp_path: Path
) -> None:
    store.initialise(_grid())
    store.commit_run(_run(TINY_EPS, INIT))
    good = _source(TINY_EPS, 300.0)

    def load_then_die(variable: str, member: int | None, step: int) -> np.ndarray | None:
        if variable == "U_10M":
            message = "simulated crash"
            raise RuntimeError(message)
        return good(variable, member, step)

    later = INIT + timedelta(hours=6)
    with pytest.raises(RuntimeError, match="simulated crash"):
        store.commit_run(_run(TINY_EPS, later, load=load_then_die))
    reopened = ProductStore.open(location=StoreLocation(root=str(tmp_path)), product=TINY_EPS)
    assert reopened.is_archived(INIT)
    assert not reopened.is_archived(later)
    assert not np.isnan(_read(reopened, "T_2M")[slot_for(TINY_EPS, INIT), 0, 0]).any()


def test_a_run_with_a_different_grid_is_reported_as_a_mismatch(store: ProductStore) -> None:
    store.initialise(_grid())
    moved = CLAT.copy()
    moved[1] += 0.5
    assert (
        store.grid_mismatch(_grid(clat=moved)) == "the cell coordinates differ from the archive's"
    )
    assert store.grid_mismatch(_grid()) is None


def test_a_run_with_a_different_cell_count_is_reported_as_a_mismatch(
    store: ProductStore,
) -> None:
    store.initialise(_grid())
    changed = replace(_grid(), n_points=len(CLAT) + 1)
    assert "cells" in (store.grid_mismatch(changed) or "")


def test_a_changed_step_list_is_a_layout_mismatch(store: ProductStore, tmp_path: Path) -> None:
    store.initialise(_grid())
    assert store.layout_mismatch() is None
    longer = replace(TINY_EPS, fields=(Field("T_2M", "T_2M", "2t", (0, 60, 120, 180)),))
    other = ProductStore.open(location=StoreLocation(root=str(tmp_path)), product=longer)
    assert other.layout_mismatch() is not None


def test_a_changed_member_count_is_a_layout_mismatch(store: ProductStore, tmp_path: Path) -> None:
    store.initialise(_grid())
    other = ProductStore.open(
        location=StoreLocation(root=str(tmp_path)), product=replace(TINY_EPS, n_members=3)
    )
    assert other.layout_mismatch() is not None


def test_xarray_opens_the_archive_with_decoded_time_coordinates(store: ProductStore) -> None:
    store.initialise(_grid())
    store.commit_run(_run(TINY_EPS, INIT))
    session = store.repository.readonly_session(branch="main")
    dataset = xr.open_zarr(session.store, consolidated=False)
    assert dataset["T_2M"].dims == ("init_time", "member", "step", "cell")
    assert np.datetime64("2026-09-25T06:00") in dataset["init_time"].values
    assert dataset["step"].values[1] == np.timedelta64(60, "m")
    assert "step_of_T_2M" in dataset


def test_a_local_repository_copied_object_by_object_to_s3_opens_identically(
    store: ProductStore, tmp_path: Path, s3_endpoint: str
) -> None:
    store.initialise(_grid())
    store.commit_run(_run(TINY_EPS, INIT))
    source_root = tmp_path / TINY_EPS.name
    client = boto3.client("s3", endpoint_url=s3_endpoint, region_name="eu-west-2")
    client.create_bucket(
        Bucket="archive", CreateBucketConfiguration={"LocationConstraint": "eu-west-2"}
    )
    for path in sorted(source_root.rglob("*")):
        if path.is_file():
            key = f"nwp/{TINY_EPS.name}/{path.relative_to(source_root).as_posix()}"
            client.put_object(Bucket="archive", Key=key, Body=path.read_bytes())
    remote = ProductStore.open(
        location=StoreLocation(root="s3://archive/nwp", s3_endpoint_url=s3_endpoint),
        product=TINY_EPS,
    )
    for name in ("T_2M", "U_10M", "status", "init_time", "clat", "step_of_U_10M"):
        np.testing.assert_array_equal(_read(store, name), _read(remote, name))
    assert remote.is_archived(INIT)
    # The copy is a working repository, not only a readable one: a new run appends to it.
    later = INIT + timedelta(hours=6)
    remote.commit_run(_run(TINY_EPS, later))
    assert remote.is_archived(later)
