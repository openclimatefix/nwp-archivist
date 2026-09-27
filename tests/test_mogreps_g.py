import httpx
import numpy as np
from support_mogreps_g import (
    BUCKET_URL,
    FIRST_COLUMN,
    FIRST_ROW,
    GRID,
    INIT,
    LATITUDE,
    LONGITUDE,
    N_COLUMNS,
    N_ROWS,
    REALIZATIONS,
    TINY_MOGREPS_G,
    FakeBucketG,
    expected_crop,
)

from nwp_archivist.dwd import Fetcher
from nwp_archivist.mogreps_g import MOGREPS_G_BASE_URL, MogrepsGSource
from nwp_archivist.products import BOX_LAT_MAX, BOX_LAT_MIN, BOX_LON_MAX, BOX_LON_MIN, MOGREPS_G
from nwp_archivist.source import Cropped, NotYet


def _source(bucket: FakeBucketG) -> MogrepsGSource:
    client = httpx.Client(transport=httpx.MockTransport(bucket.handle))
    fetcher = Fetcher(client=client, sleep=lambda _seconds: None, jitter=lambda: 0.0)
    return MogrepsGSource(fetcher=fetcher, base_url=BUCKET_URL)


def test_box_rectangle_keeps_only_cells_inside_the_box() -> None:
    rows, columns = GRID.shape or (0, 0)
    assert 0 < rows < N_ROWS
    assert 0 < columns < N_COLUMNS
    lat = LATITUDE[FIRST_ROW : FIRST_ROW + rows]
    lon = LONGITUDE[FIRST_COLUMN : FIRST_COLUMN + columns]
    assert lat.min() >= BOX_LAT_MIN
    assert lat.max() <= BOX_LAT_MAX
    assert lon.min() >= BOX_LON_MIN
    assert lon.max() <= BOX_LON_MAX


def test_box_rectangle_is_the_smallest_rectangle_that_contains_every_hit() -> None:
    # The row and column just outside the kept rectangle, on every side, must fall outside the box.
    rows, columns = GRID.shape or (0, 0)
    if FIRST_ROW > 0:
        assert not (BOX_LAT_MIN <= LATITUDE[FIRST_ROW - 1] <= BOX_LAT_MAX)
    if FIRST_ROW + rows < N_ROWS:
        assert not (BOX_LAT_MIN <= LATITUDE[FIRST_ROW + rows] <= BOX_LAT_MAX)
    if FIRST_COLUMN > 0:
        assert not (BOX_LON_MIN <= LONGITUDE[FIRST_COLUMN - 1] <= BOX_LON_MAX)
    if FIRST_COLUMN + columns < N_COLUMNS:
        assert not (BOX_LON_MIN <= LONGITUDE[FIRST_COLUMN + columns] <= BOX_LON_MAX)


def test_default_base_url_points_at_the_global_ensemble_bucket() -> None:
    assert "met-office-global-ensemble-model-data" in MOGREPS_G_BASE_URL
    assert MOGREPS_G_BASE_URL.endswith("/global-ensemble")


def test_fetch_grid_reads_a_regular_lat_lon_grid_not_a_projection() -> None:
    bucket = FakeBucketG()
    source = _source(bucket)
    grid = source.fetch_grid(product=TINY_MOGREPS_G, init_time=INIT, only=None, previous=None)
    assert not isinstance(grid, NotYet)
    assert grid.shape == GRID.shape
    np.testing.assert_array_equal(grid.cell_index, GRID.cell_index)
    np.testing.assert_allclose(grid.statics["clat"], GRID.statics["clat"])
    np.testing.assert_allclose(grid.statics["clon"], GRID.statics["clon"])


def test_fetch_crops_a_surface_field_by_byte_range() -> None:
    bucket = FakeBucketG()
    source = _source(bucket)
    grid = source.fetch_grid(product=TINY_MOGREPS_G, init_time=INIT, only=None, previous=None)
    assert not isinstance(grid, NotYet)
    (file,) = [
        f
        for f in source.expected_files(TINY_MOGREPS_G, INIT)
        if f.field.variable == "sw" and f.step_minutes == 0 and f.member == 1
    ]
    outcome = source.fetch(file, grid)
    assert isinstance(outcome, Cropped)
    np.testing.assert_allclose(outcome.values, expected_crop(file))
    assert outcome.member_ids == REALIZATIONS
    # A regular lat/lon grid is far smaller than 128x128 per chunk here, so the whole tiny file's
    # one chunk row is fetched, never the full file: only one distinct range request per member.
    assert len(bucket.requests) < N_ROWS * N_COLUMNS


def test_fetch_crops_a_height_level_field_and_selects_the_right_member() -> None:
    bucket = FakeBucketG()
    source = _source(bucket)
    grid = source.fetch_grid(product=TINY_MOGREPS_G, init_time=INIT, only=None, previous=None)
    assert not isinstance(grid, NotYet)
    (member_1,) = [
        f
        for f in source.expected_files(TINY_MOGREPS_G, INIT)
        if f.field.variable == "wind_100m" and f.step_minutes == 60 and f.member == 1
    ]
    (member_3,) = [
        f
        for f in source.expected_files(TINY_MOGREPS_G, INIT)
        if f.field.variable == "wind_100m" and f.step_minutes == 60 and f.member == 3
    ]
    outcome_1 = source.fetch(member_1, grid)
    outcome_3 = source.fetch(member_3, grid)
    assert isinstance(outcome_1, Cropped)
    assert isinstance(outcome_3, Cropped)
    np.testing.assert_allclose(outcome_1.values, expected_crop(member_1))
    np.testing.assert_allclose(outcome_3.values, expected_crop(member_3))
    assert not np.allclose(outcome_1.values, outcome_3.values)


def test_the_real_product_lists_one_file_per_member_variable_and_lead_time() -> None:
    files = MogrepsGSource(fetcher=Fetcher(client=httpx.Client())).expected_files(MOGREPS_G, INIT)
    n_steps = sum(len(f.steps_minutes) for f in MOGREPS_G.fields)
    assert len(files) == n_steps * len(MOGREPS_G.members)
    assert all("global-ensemble" in f.url for f in files)
