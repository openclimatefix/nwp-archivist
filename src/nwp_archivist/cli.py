"""The `archive-record` command: run one recording cycle and exit.

A systemd timer runs the command every 15 minutes. Each run logs one line per examined run to
standard output, where journald captures it.
"""

import argparse
import logging
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Final

import httpx
import icechunk

from nwp_archivist.dwd import Fetcher
from nwp_archivist.products import PRODUCTS
from nwp_archivist.recorder import DEFAULT_MIN_FREE_BYTES, Recorder, RecorderConfig
from nwp_archivist.reporting import make_reporter
from nwp_archivist.store import StoreLocation

DEFAULT_CACHE_DIR: Final[str] = "/mnt/data/nwp-archive-cache"
REQUEST_TIMEOUT_SECONDS: Final[float] = 60.0


def build_parser() -> argparse.ArgumentParser:
    """Define the command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="archive-record",
        description="Record one cycle of DWD ensemble weather products.",
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
        default=sorted(PRODUCTS),
        help="The products to record (default: all).",
    )
    parser.add_argument("--workers", type=int, default=8, help="Files downloaded in parallel.")
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
    config = RecorderConfig(
        store=StoreLocation(
            root=args.store_root,
            s3_endpoint_url=os.environ.get("ARCHIVE_S3_ENDPOINT_URL"),
        ),
        cache_dir=Path(args.cache_dir),
        min_free_bytes=int(args.min_free_gib * 1024**3),
        workers=args.workers,
    )
    limits = httpx.Limits(max_connections=args.workers, max_keepalive_connections=args.workers)
    with httpx.Client(timeout=REQUEST_TIMEOUT_SECONDS, limits=limits) as client:
        recorder = Recorder(config=config, fetcher=Fetcher(client=client), reporter=make_reporter())
        recorder.run_cycle([PRODUCTS[name] for name in args.products])
    return 0
