"""Fetch real fields from DWD and the Met Office. Run with `uv run pytest --run-network`."""

from datetime import UTC, datetime, timedelta

import httpx
import numpy as np
import pytest

from nwp_archivist.dwd import Decoded, Expectation, Fetcher
from nwp_archivist.mogreps import MogrepsSource
from nwp_archivist.products import (
    ICON_D2,
    ICON_D2_EPS,
    ICON_EU_EPS,
    MOGREPS_UK,
    Product,
    expected_files,
    static_urls,
)
from nwp_archivist.source import Cropped, NotYet
from nwp_archivist.store import box_cell_index

pytestmark = pytest.mark.network


def _recent_run(product: Product) -> datetime:
    """A run that finished publishing, at least 9 hours old and inside the provider's retention."""
    target = datetime.now(UTC) - timedelta(hours=9 if product.source == "dwd" else 48)
    return target.replace(minute=0, second=0, microsecond=0) - timedelta(
        hours=target.hour % product.cycle_hours
    )


@pytest.mark.parametrize("product", [ICON_D2_EPS, ICON_EU_EPS, ICON_D2], ids=lambda p: p.name)
def test_a_real_temperature_field_decodes_and_is_in_range(product: Product) -> None:
    init_time = _recent_run(product)
    (file,) = [
        f
        for f in expected_files(product, init_time)
        if f.field.variable == "T_2M" and f.step_minutes == 360 and f.member in (None, 1)
    ]
    with httpx.Client(timeout=60.0) as client:
        outcome = Fetcher(client=client).fetch(
            file.url,
            expect=Expectation(short_name="2t", step_minutes=360, member=file.member, level=None),
        )
        assert isinstance(outcome, Decoded)
        clat = Fetcher(client=client).fetch(
            static_urls(product, init_time)["clat"], expect=Expectation()
        )
        clon = Fetcher(client=client).fetch(
            static_urls(product, init_time)["clon"], expect=Expectation()
        )
    assert isinstance(clat, Decoded)
    assert isinstance(clon, Decoded)
    assert outcome.values.shape == clat.values.shape
    kept = box_cell_index(clat=clat.values, clon=clon.values)
    assert len(kept) > 1000
    temperature = outcome.values[kept]
    assert np.isfinite(temperature).mean() > 0.9
    assert 230.0 < np.nanmin(temperature) < np.nanmax(temperature) < 330.0


def test_a_real_mogreps_wind_and_temperature_field_are_cropped_and_in_range() -> None:
    init_time = _recent_run(MOGREPS_UK)
    with httpx.Client(timeout=60.0) as client:
        source = MogrepsSource(fetcher=Fetcher(client=client))
        grid = source.fetch_grid(product=MOGREPS_UK, init_time=init_time, only=None, previous=None)
        assert not isinstance(grid, NotYet)
        assert grid.shape == (707, 494)
        files = source.expected_files(MOGREPS_UK, init_time)
        for variable, low, high in (
            ("wind_speed_100m", 0.0, 80.0),
            ("temperature_1p5m", 230.0, 330.0),
        ):
            (file,) = [
                f
                for f in files
                if f.field.variable == variable and f.step_minutes == 360 and f.member == 2
            ]
            result = source.fetch(file, grid)
            assert isinstance(result, Cropped)
            assert result.values.shape == (707 * 494,)
            assert np.isfinite(result.values).mean() > 0.99
            assert low <= np.nanmin(result.values) < np.nanmax(result.values) <= high
            assert len(result.member_ids) == 3
