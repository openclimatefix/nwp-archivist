"""Coordinating backfill between recorders that share a provider's retention window.

MOGREPS-UK and MOGREPS-G each run as their own systemd service with their own cache directory, so
each recorder decides its backfill slice with no view of the other. Both compete for the same
scarce thing: the days left before a run's oldest still-fetchable file falls out of the provider's
retention window. Before spending a cycle's backfill slice, a recorder writes its own reading of
that to a small shared JSON file, then reads the other product's most recent reading and skips
backfill this cycle (the live pass still runs) when the other recorder is meaningfully closer to
losing a run. Nothing here raises: a missing, stale, or corrupt file is treated as no signal, and
backfill goes ahead as if the file did not exist.
"""

import json
import math
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

# An entry older than this is presumed to be from a recorder that is no longer running (or is stuck
# well short of a cycle), so it no longer says anything about how urgent that product's backfill is.
STALE_AFTER: Final[timedelta] = timedelta(hours=1)
# Skip this cycle's backfill only when the other product is closer to losing a run by more than
# this many days, so that two products of similar urgency both still get a slice most cycles.
YIELD_MARGIN_DAYS: Final[float] = 0.5


def _write_atomically(path: Path, data: bytes) -> None:
    """Write bytes so that a crash leaves either no file or the whole file, never a part."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        handle.write(data)
    temporary.replace(path)


def _read_all(path: Path) -> dict[str, dict[str, object]]:
    """Every product's last entry, or `{}` if absent, unreadable, or not a JSON object."""
    try:
        parsed = json.loads(path.read_text())
    except FileNotFoundError, OSError, ValueError:
        return {}
    if not isinstance(parsed, dict):
        return {}
    return parsed


def write_urgency(path: Path, *, product: str, days_left: float, now: datetime) -> None:
    """Record `product`'s current backfill urgency, keeping every other product's last entry.

    Args:
        path: The shared urgency file.
        product: The product recording its urgency.
        days_left: Days until the oldest run still in this product's backfill queue falls out of
            the provider's retention window, or `math.inf` when nothing is queued for backfill.
        now: The time this entry was written.
    """
    entries = _read_all(path)
    entries[product] = {"days_left": days_left, "written_at": now.isoformat()}
    _write_atomically(path, json.dumps(entries).encode())


def should_yield(path: Path, *, product: str, my_days_left: float, now: datetime) -> bool:
    """Whether `product` should skip this cycle's backfill slice for a more urgent other product.

    Args:
        path: The shared urgency file.
        product: The product deciding whether to yield.
        my_days_left: This product's own days left before retention, from the same reading that
            was (or will be) passed to `write_urgency`.
        now: The current time, used to reject a stale entry.

    Returns:
        Whether another product's fresh, readable entry is more urgent than `my_days_left` by more
        than `YIELD_MARGIN_DAYS`. Never raises: a missing, stale, or corrupt file yields `False`.
    """
    for other_product, entry in _read_all(path).items():
        if other_product == product:
            continue
        try:
            days_left = float(entry["days_left"])  # ty: ignore[invalid-argument-type]
            written_at = datetime.fromisoformat(str(entry["written_at"]))
        except KeyError, TypeError, ValueError:
            continue
        if written_at.tzinfo is None:
            written_at = written_at.replace(tzinfo=UTC)
        if now - written_at > STALE_AFTER:
            continue
        if days_left < my_days_left - YIELD_MARGIN_DAYS:
            return True
    return False


def oldest_run_days_left(
    *, oldest_backfill_run: datetime | None, lookback_hours: float, now: datetime
) -> float:
    """Days until the oldest run still queued for backfill falls out of the retention window.

    Args:
        oldest_backfill_run: The oldest run still waiting to be backfilled, or `None` when nothing
            is queued.
        lookback_hours: How far back the provider's retention window reaches.
        now: The current time.

    Returns:
        `math.inf` when nothing is queued for backfill, otherwise the days left, which may be
        negative for a run the provider has likely already deleted.
    """
    if oldest_backfill_run is None:
        return math.inf
    retention_ends = oldest_backfill_run + timedelta(hours=lookback_hours)
    return (retention_ends - now).total_seconds() / 86400
