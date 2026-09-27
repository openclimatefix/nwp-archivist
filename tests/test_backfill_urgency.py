"""Tests for the shared backfill-urgency file two recorders use to yield a slice to each other."""

import math
from datetime import UTC, datetime, timedelta
from pathlib import Path

from nwp_archivist.backfill_urgency import (
    oldest_run_days_left,
    should_yield,
    write_urgency,
)

NOW = datetime(2026, 9, 27, 12, 0, tzinfo=UTC)


def test_yields_when_the_other_product_is_meaningfully_more_urgent(tmp_path: Path) -> None:
    """A fresh entry more than the margin more urgent than mine makes this product yield."""
    path = tmp_path / "urgency.json"
    write_urgency(path, product="mogreps-uk", days_left=1.0, now=NOW)
    assert should_yield(path, product="mogreps-g", my_days_left=3.0, now=NOW) is True


def test_does_not_yield_within_the_margin(tmp_path: Path) -> None:
    """An other product only slightly more urgent, within the margin, does not cause a yield."""
    path = tmp_path / "urgency.json"
    write_urgency(path, product="mogreps-uk", days_left=2.6, now=NOW)
    assert should_yield(path, product="mogreps-g", my_days_left=3.0, now=NOW) is False


def test_does_not_yield_when_i_am_the_more_urgent_one(tmp_path: Path) -> None:
    """A product that is itself the more urgent one never yields to a calmer other product."""
    path = tmp_path / "urgency.json"
    write_urgency(path, product="mogreps-uk", days_left=10.0, now=NOW)
    assert should_yield(path, product="mogreps-g", my_days_left=1.0, now=NOW) is False


def test_a_missing_file_never_yields(tmp_path: Path) -> None:
    """No file at all is the same as no signal: proceed with backfill."""
    path = tmp_path / "does-not-exist.json"
    assert should_yield(path, product="mogreps-g", my_days_left=1.0, now=NOW) is False


def test_a_corrupt_file_never_yields(tmp_path: Path) -> None:
    """Unparseable JSON is treated as no signal, never as a reason to raise or to yield."""
    path = tmp_path / "urgency.json"
    path.write_text("not json")
    assert should_yield(path, product="mogreps-g", my_days_left=1.0, now=NOW) is False


def test_a_stale_entry_never_yields(tmp_path: Path) -> None:
    """An entry older than the staleness window no longer says anything about urgency."""
    path = tmp_path / "urgency.json"
    write_urgency(path, product="mogreps-uk", days_left=0.1, now=NOW - timedelta(hours=2))
    assert should_yield(path, product="mogreps-g", my_days_left=3.0, now=NOW) is False


def test_own_entry_is_never_compared_against_itself(tmp_path: Path) -> None:
    """A product's own previous entry, still under its own name, is skipped when reading others."""
    path = tmp_path / "urgency.json"
    write_urgency(path, product="mogreps-g", days_left=0.1, now=NOW)
    assert should_yield(path, product="mogreps-g", my_days_left=3.0, now=NOW) is False


def test_a_non_object_json_file_never_yields(tmp_path: Path) -> None:
    """Valid JSON that is not an object (a list, null, a bare number) is treated as no signal."""
    path = tmp_path / "urgency.json"
    for body in ("[]", "null", "3"):
        path.write_text(body)
        assert should_yield(path, product="mogreps-g", my_days_left=1.0, now=NOW) is False


def test_write_urgency_over_a_non_object_json_file_does_not_raise(tmp_path: Path) -> None:
    """Writing over a file that holds valid JSON of the wrong shape replaces it, without raising."""
    path = tmp_path / "urgency.json"
    path.write_text("[]")
    write_urgency(path, product="mogreps-g", days_left=1.0, now=NOW)
    assert should_yield(path, product="mogreps-uk", my_days_left=5.0, now=NOW) is True


def test_write_urgency_keeps_the_other_products_entry(tmp_path: Path) -> None:
    """Writing one product's entry never overwrites another product's entry in the same file."""
    path = tmp_path / "urgency.json"
    write_urgency(path, product="mogreps-uk", days_left=1.0, now=NOW)
    write_urgency(path, product="mogreps-g", days_left=5.0, now=NOW)
    assert should_yield(path, product="mogreps-g", my_days_left=5.0, now=NOW) is True


def test_oldest_run_days_left_is_infinite_with_nothing_queued() -> None:
    """No run waiting for backfill is the least urgent state there is."""
    days_left = oldest_run_days_left(oldest_backfill_run=None, lookback_hours=720.0, now=NOW)
    assert math.isinf(days_left)


def test_oldest_run_days_left_counts_down_to_retention() -> None:
    """A run 29 days old with 30 days of retention has about one day left."""
    oldest = NOW - timedelta(days=29)
    days_left = oldest_run_days_left(oldest_backfill_run=oldest, lookback_hours=30 * 24.0, now=NOW)
    assert days_left == 1.0
