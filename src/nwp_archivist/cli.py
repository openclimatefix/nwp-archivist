"""The `archive-record` command: run one recording cycle and exit.

A systemd timer runs the command every 15 minutes. Each run logs one line per examined run to
standard output, where journald captures it.
"""

import argparse
import fcntl
import logging
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Final

import httpx
import icechunk

from nwp_archivist.dwd import Fetcher
from nwp_archivist.mogreps import MogrepsSource
from nwp_archivist.products import PRODUCTS
from nwp_archivist.recorder import DEFAULT_MIN_FREE_BYTES, Recorder, RecorderConfig
from nwp_archivist.reporting import make_reporter
from nwp_archivist.store import StoreLocation

DEFAULT_CACHE_DIR: Final[str] = "/mnt/data/nwp-archive-cache"
REQUEST_TIMEOUT_SECONDS: Final[float] = 60.0
LOCK_FILE_NAME: Final[str] = ".lock"


def build_parser() -> argparse.ArgumentParser:
    """Define the command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="archive-record",
        description="Record one cycle of short-retention ensemble weather products.",
    )
    parser.add_argument(
        "--store-root",
        default=os.environ.get("ARCHIVE_STORE_ROOT"),
        help=(
            "Where the product repositories live: a local directory or an s3:// prefix "
            "(default: the ARCHIVE_STORE_ROOT environment variable)."
        ),
    )
    parser.add_argument(
        "--cache-dir",
        default=DEFAULT_CACHE_DIR,
        help="Where cropped files are cached until their run is committed.",
    )
    parser.add_argument(
        "--products",
        nargs="+",
        choices=sorted(PRODUCTS),
        default=sorted(name for name, product in PRODUCTS.items() if product.source == "dwd"),
        help="The products to record (default: the DWD products).",
    )
    parser.add_argument(
        "--monitor-slug",
        default=None,
        help=(
            "The Sentry cron monitor for this service (default: nwp-archive-<provider> when every "
            "product has the same provider source, such as nwp-archive-dwd, otherwise "
            "nwp-archive-record)."
        ),
    )
    parser.add_argument(
        "--fetch-processes",
        type=int,
        default=1,
        help="Worker processes that fetch MOGREPS-UK files (1 means threads in this process).",
    )
    parser.add_argument("--max-backfill-runs", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--workers", type=int, default=8, help="Files downloaded in parallel.")
    parser.add_argument(
        "--backfill-minutes",
        type=float,
        default=15.0,
        help="How long a cycle may spend on runs older than a product's live window.",
    )
    parser.add_argument(
        "--min-free-gib",
        type=float,
        default=DEFAULT_MIN_FREE_BYTES / 1024**3,
        help="Report a fault when the cache disk has less free space than this.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run one recording cycle.

    Args:
        argv: The command-line arguments, or `None` to read `sys.argv`.

    Returns:
        The process exit code: 0 after a cycle (whatever it found, because a fault is reported
        through the reporter), 2 when the store root is not configured.
    """
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    for noisy in ("httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    # Icechunk logs to standard error from Rust; keep only its errors.
    icechunk.set_logs_filter("error")
    if not args.store_root:
        logging.getLogger(__name__).error("set --store-root or ARCHIVE_STORE_ROOT")
        return 2
    sources = {PRODUCTS[name].source for name in args.products}
    monitor_slug = args.monitor_slug or (
        f"nwp-archive-{sources.pop()}" if len(sources) == 1 else "nwp-archive-record"
    )
    config = RecorderConfig(
        store=StoreLocation(
            root=args.store_root,
            s3_endpoint_url=os.environ.get("ARCHIVE_S3_ENDPOINT_URL"),
        ),
        cache_dir=Path(args.cache_dir),
        min_free_bytes=int(args.min_free_gib * 1024**3),
        workers=args.workers,
        backfill_seconds=args.backfill_minutes * 60,
        fetch_processes=args.fetch_processes,
        max_backfill_runs=args.max_backfill_runs,
    )
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    # Two cycles at once could lose an Icechunk commit on local storage and corrupt cache files, so
    # only one process runs at a time. The lock is released when the process exits.
    with (cache_dir / LOCK_FILE_NAME).open("w") as lock_file:
        try:
            fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            logging.getLogger(__name__).info(
                "another archive-record run holds the lock on %s; exiting", cache_dir
            )
            return 0
        limits = httpx.Limits(max_connections=args.workers, max_keepalive_connections=args.workers)
        with httpx.Client(timeout=REQUEST_TIMEOUT_SECONDS, limits=limits) as client:
            fetcher = Fetcher(client=client)
            recorder = Recorder(
                config=config,
                fetcher=fetcher,
                reporter=make_reporter(monitor_slug=monitor_slug),
                mogreps_source=MogrepsSource(fetcher=fetcher),
            )
            try:
                recorder.run_cycle([PRODUCTS[name] for name in args.products])
            finally:
                recorder.close()
    return 0
