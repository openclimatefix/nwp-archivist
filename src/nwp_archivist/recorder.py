"""The recording cycle: work out which runs should exist, fetch their files, and commit them.

Nothing here raises because a provider file is absent or late. A run is retried every cycle, and
once its deadline passes it is committed with whatever arrived, marked `partial`, or marked
`missing` if nothing arrived. Each run sits in its own `try` block, so one failing run never stops
the others.
"""

import logging
import shutil
import subprocess
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Final

import numpy as np

from nwp_archivist.cache import ProductCache, RunCache, RunGrid
from nwp_archivist.dwd import DwdSource, Fetcher
from nwp_archivist.products import DWD_BASE_URL, ExpectedFile, Product, run_init_times
from nwp_archivist.reporting import FaultType, Reporter
from nwp_archivist.source import NotYet, Source
from nwp_archivist.store import (
    STATUS_COMPLETE,
    STATUS_MISSING,
    STATUS_NAMES,
    STATUS_PARTIAL,
    ProductStore,
    RunToCommit,
    StoreLocation,
    slot_for,
)

logger = logging.getLogger(__name__)

DEFAULT_MIN_FREE_BYTES: Final[int] = 10 * 1024**3

# DWD keeps a run for about 24 to 27 hours, so a run older than this has nothing left to fetch.
DEFAULT_LOOKBACK_HOURS: Final[float] = 27.0

# A run is normally fully published within three hours of its start delay. From then on, every
# missing file is tried each cycle, so that a single file DWD skipped does not hide the files after
# it in its sequence until the deadline.
EXHAUSTIVE_AFTER_START_HOURS: Final[float] = 3.0

# At the deadline, a pass that saw errors other than 404 is repeated this many times before the
# run is committed, so that one network blip does not commit a run as `partial`.
DEADLINE_RETRY_PASSES: Final[int] = 2

_GIT_TIMEOUT_SECONDS: Final[float] = 5.0

# A run that no pass found any file of is looked at again after this delay, which doubles with each
# empty pass up to the cap, so that a run the provider never published costs a few requests a day.
BACKOFF_FIRST: Final[timedelta] = timedelta(minutes=30)
BACKOFF_CAP: Final[timedelta] = timedelta(hours=12)


class GridChangedError(Exception):
    """A run's grid, member count, or step list differs from the archive's."""


def code_version() -> str:
    """The package version with the git commit hash of the checkout, when there is one.

    The recorder runs from a working tree, so the hash is what says which code recorded a run.
    Outside a git checkout the version alone is returned, or `unknown` when it is not installed.
    """
    try:
        package_version = version("nwp-archivist")
    except PackageNotFoundError:
        package_version = "unknown"
    try:
        completed = subprocess.run(
            ["git", "-C", str(Path(__file__).parent), "rev-parse", "--short=12", "HEAD"],
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT_SECONDS,
            check=False,
        )
    except OSError, subprocess.SubprocessError:
        return package_version
    commit = completed.stdout.strip()
    if completed.returncode != 0 or not commit:
        return package_version
    return f"{package_version}+{commit}"


@dataclass(frozen=True)
class RecorderConfig:
    """The settings of a recording cycle.

    Attributes:
        store: Where the product repositories live.
        cache_dir: Where cropped files are cached until their run is committed.
        min_free_bytes: The free disk space under which a cycle reports a `disk_low` fault.
        lookback_hours: How far back from now a cycle looks for runs that are not yet archived.
            Runs that still have a cache directory are always examined as well.
        workers: The number of files downloaded in parallel.
        base_url: The root of DWD's directory layout.
        backfill_seconds: How long a cycle may spend on runs that are not live (see
            `Product.live_hours`) once its live runs are done. A run in progress when the time is
            up stops at its next file.
        backfill_workers: The number of files downloaded in parallel for a run that is not live,
            which keeps the backfill's request rate modest.
    """

    store: StoreLocation
    cache_dir: Path
    min_free_bytes: int = DEFAULT_MIN_FREE_BYTES
    lookback_hours: float = DEFAULT_LOOKBACK_HOURS
    workers: int = 8
    base_url: str = DWD_BASE_URL
    backfill_seconds: float = 15 * 60.0
    backfill_workers: int = 2


@dataclass
class _FetchPassResult:
    """What one fetch pass over a run's files found."""

    generating_process: int | None = None
    grid_changed: str | None = None
    n_fetched: int = 0
    transient: bool = False
    member_ids: tuple[int, ...] = ()


@dataclass(frozen=True)
class _RunOutcome:
    """What one cycle did with one run."""

    clean: bool
    n_fetched: int = 0
    empty: bool = False


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _slot_archived(product: Product, statuses: np.ndarray, init_time: datetime) -> bool:
    """Whether an already-read status array holds this run."""
    slot = slot_for(product, init_time)
    return slot < len(statuses) and statuses[slot] != 0


class _FaultLedger:
    """Reports a persistent fault once, until a cycle passes without it.

    A fault that lasts, such as a repository that cannot be reached, would otherwise raise an event
    every cycle. The active faults are kept on disk, because each cycle is a new process.
    """

    def __init__(self, *, cache: ProductCache, reporter: Reporter) -> None:
        self._cache = cache
        self._reporter = reporter
        self._previous = cache.load_active_faults()
        self._current: set[str] = set()

    def report(
        self, *, product: str, init_time: datetime | None, fault: FaultType, detail: str
    ) -> None:
        """Send the fault unless the previous cycle already reported it."""
        key = f"{fault}:{init_time.isoformat() if init_time else '-'}"
        self._current.add(key)
        if key not in self._previous:
            self._reporter.fault(product=product, init_time=init_time, fault=fault, detail=detail)

    def save(self) -> None:
        """Remember which faults are active, so that the faults that cleared can fire again."""
        self._cache.save_active_faults(self._current)


class Recorder:
    """Runs recording cycles for a set of products."""

    def __init__(
        self,
        *,
        config: RecorderConfig,
        fetcher: Fetcher,
        reporter: Reporter,
        clock: Callable[[], datetime] = _utc_now,
        mogreps_source: Source | None = None,
    ) -> None:
        """Build a recorder.

        Args:
            config: The recorder's settings.
            fetcher: The DWD file downloader.
            reporter: Where faults and the cycle check-in go.
            clock: Returns the current time, replaceable so that tests control time.
            mogreps_source: Reads MOGREPS-UK files, if that product is recorded.
        """
        self.config = config
        self.reporter = reporter
        self.clock = clock
        self._sources: dict[str, Source] = {
            "dwd": DwdSource(fetcher=fetcher, base_url=config.base_url)
        }
        if mogreps_source is not None:
            self._sources["mogreps"] = mogreps_source
        self._code_version = code_version()

    def run_cycle(self, products: Sequence[Product]) -> bool:
        """Run one recording cycle over `products`.

        Args:
            products: The products to record.

        Returns:
            Whether every product's cycle finished without a fault, which is also what the cron
            check-in reports.
        """
        clean = True
        for product in products:
            clean &= self._run_product(product)
        self.reporter.check_in(ok=clean)
        return clean

    def _run_product(self, product: Product) -> bool:
        """Run one product's cycle, returning whether it finished without a fault."""
        cache = ProductCache(root=self.config.cache_dir, product=product.name)
        ledger = _FaultLedger(cache=cache, reporter=self.reporter)
        try:
            return self._run_product_with(product=product, cache=cache, ledger=ledger)
        finally:
            ledger.save()

    def _run_product_with(
        self, *, product: Product, cache: ProductCache, ledger: _FaultLedger
    ) -> bool:
        try:
            store = ProductStore.open(location=self.config.store, product=product)
            mismatch = store.layout_mismatch()
            archived = store.statuses()
        except Exception as error:
            logger.warning("cannot open the %s repository", product.name, exc_info=True)
            ledger.report(
                product=product.name,
                init_time=None,
                fault="commit_failed",
                detail=f"cannot open the repository: {error!r}",
            )
            return False
        if mismatch is not None:
            self._halt(cache=cache, product=product, reason=mismatch)
        now = self.clock()
        clean = self._check_free_space(product=product, ledger=ledger)
        window = run_init_times(
            product,
            first=now - timedelta(hours=product.lookback_hours or self.config.lookback_hours),
            last=now - timedelta(hours=product.start_delay_hours),
        )
        runs = sorted({*window, *cache.cached_runs()})
        live_runs = [
            run
            for run in runs
            if product.live_hours is None or now - run <= timedelta(hours=product.live_hours)
        ]
        # A backfill starts with the newest run, because the provider deletes the oldest first.
        backfill_runs = sorted(set(runs) - set(live_runs), reverse=True)
        backfill_end: float | None = None
        oldest_fetched: datetime | None = None
        for init_time in (*live_runs, *backfill_runs):
            backfill = init_time not in live_runs
            if backfill and backfill_end is None:
                # The budget covers the backfill only, so its clock starts after the live runs.
                backfill_end = time.monotonic() + self.config.backfill_seconds
            if backfill_end is not None and backfill and time.monotonic() >= backfill_end:
                break
            if _slot_archived(product, archived, init_time) or cache.is_done(init_time):
                # A commit can land without its cleanup, so a leftover cache is deleted here.
                cache.run(init_time).delete()
                continue
            if self._backing_off(cache.run(init_time), now=now):
                continue
            try:
                outcome = self._process_run(
                    product=product,
                    store=store,
                    cache=cache,
                    ledger=ledger,
                    init_time=init_time,
                    now=now,
                    stop=(lambda end=backfill_end: time.monotonic() >= end)
                    if backfill_end
                    else None,
                )
            except Exception as error:
                logger.warning(
                    "run %s %s failed", product.name, init_time.isoformat(), exc_info=True
                )
                ledger.report(
                    product=product.name,
                    init_time=init_time,
                    fault="run_error",
                    detail=repr(error),
                )
                clean = False
                continue
            clean &= outcome.clean
            if outcome.empty and product.missing_after_hours is not None:
                self._record_empty_pass(cache.run(init_time), now=now)
            if outcome.n_fetched and oldest_fetched is None:
                oldest_fetched = init_time
        if oldest_fetched is not None:
            # DWD's real retention is not known exactly, so the age of the oldest run we could
            # still download shows how much margin the deadline has.
            logger.info(
                "product=%s oldest_fetched_age_hours=%.1f",
                product.name,
                (now - oldest_fetched).total_seconds() / 3600,
            )
        return clean

    @staticmethod
    def _backing_off(run_cache: RunCache, *, now: datetime) -> bool:
        """Whether an earlier empty pass put this run off until later."""
        backoff = run_cache.load_backoff()
        return backoff is not None and now < backoff[1]

    @staticmethod
    def _record_empty_pass(run_cache: RunCache, *, now: datetime) -> None:
        """Delay the next look at a run with no files, doubling the delay each time."""
        backoff = run_cache.load_backoff()
        attempts = 0 if backoff is None else backoff[0]
        run_cache.save_backoff(
            attempts=attempts + 1, next_try=now + min(BACKOFF_FIRST * 2**attempts, BACKOFF_CAP)
        )

    def _halt(
        self,
        *,
        cache: ProductCache,
        product: Product,
        reason: str,
        init_time: datetime | None = None,
    ) -> None:
        """Stop commits for a product, and report it the first time. Fetching goes on."""
        if cache.halted() is not None:
            return
        cache.halt(reason)
        self.reporter.fault(
            product=product.name,
            init_time=init_time,
            fault="grid_changed",
            detail=reason,
        )

    def _process_run(
        self,
        *,
        product: Product,
        store: ProductStore,
        cache: ProductCache,
        ledger: _FaultLedger,
        init_time: datetime,
        now: datetime,
        stop: Callable[[], bool] | None = None,
    ) -> _RunOutcome:
        """Fetch what is missing of one run and commit it if it is complete or past its deadline.

        Args:
            product: The product being recorded.
            store: The product's repository.
            cache: The product's local cache.
            ledger: Where persistent faults are reported.
            init_time: The run's initialisation time.
            now: The time this cycle started.
            stop: Says when the fetch pass should give up on the rest of the run, or `None`.

        Returns:
            Whether the run was handled without a fault, how many files were downloaded, and
            whether the run is still waiting with none of its files.
        """
        source = self._sources[product.source]
        files = source.expected_files(product, init_time)
        run_cache = cache.run(init_time)
        past_deadline = now >= product.deadline(init_time)
        past_missing = now >= product.missing_deadline(init_time)
        exhaustive = now >= init_time + timedelta(
            hours=product.start_delay_hours + EXHAUSTIVE_AFTER_START_HOURS
        )
        grid, mismatch = self._ensure_grid(
            source=source,
            product=product,
            store=store,
            run_cache=run_cache,
            ledger=ledger,
            init_time=init_time,
            past_deadline=past_missing,
        )
        # A halted product keeps fetching, so that no run is lost while a person decides what to do.
        # Once a run is past its deadline nothing more will arrive, so it is left alone.
        halted = cache.halted() is not None
        n_fetched = 0
        received = 0
        if grid is not None:
            if not (halted and past_deadline):
                n_fetched, changed = self._fetch_run(
                    source=source,
                    files=files,
                    grid=grid,
                    run_cache=run_cache,
                    exhaustive=exhaustive,
                    past_deadline=past_deadline,
                    stop=stop,
                )
                mismatch = mismatch or changed
            received = run_cache.count_received(files, n_cells=len(grid.cell_index))
        if mismatch is not None:
            self._halt(cache=cache, product=product, reason=mismatch, init_time=init_time)
        if received == len(files):
            status = STATUS_COMPLETE
        elif stop is not None and stop():
            # The backfill's time budget ended mid-run. The rest of the files are still in the
            # bucket, so the run waits for the next cycle instead of committing partial.
            self._log_run(product, init_time, len(files), received, "waiting")
            return _RunOutcome(clean=True, n_fetched=n_fetched)
        elif received > 0 and past_deadline:
            status = STATUS_PARTIAL
        elif received == 0 and past_missing:
            status = STATUS_MISSING
        else:
            self._log_run(product, init_time, len(files), received, "waiting")
            return _RunOutcome(clean=True, n_fetched=n_fetched, empty=received == 0)
        if cache.halted() is not None:
            self._log_run(product, init_time, len(files), received, "halted")
            return _RunOutcome(clean=False, n_fetched=n_fetched)
        self._log_run(product, init_time, len(files), received, STATUS_NAMES[status])
        if grid is None:
            # Nothing can be written without a grid, and a first run has none to write.
            self.reporter.fault(
                product=product.name,
                init_time=init_time,
                fault="missing",
                detail="no grid files arrived and the archive holds no grid, so nothing was stored",
            )
            cache.mark_done(init_time)
            run_cache.delete()
            return _RunOutcome(clean=False, n_fetched=n_fetched)
        committed = self._commit(
            product=product,
            store=store,
            run_cache=run_cache,
            ledger=ledger,
            init_time=init_time,
            grid=grid,
            status=status,
            n_files=len(files),
            received=received,
        )
        return _RunOutcome(clean=committed, n_fetched=n_fetched)

    def _commit(
        self,
        *,
        product: Product,
        store: ProductStore,
        run_cache: RunCache,
        ledger: _FaultLedger,
        init_time: datetime,
        grid: RunGrid,
        status: int,
        n_files: int,
        received: int,
    ) -> bool:
        """Commit one run, and only after the commit succeeds delete its local files."""
        try:
            store.initialise(grid)
            store.commit_run(
                RunToCommit(
                    init_time=init_time,
                    status=status,
                    files_expected=n_files,
                    files_received=received,
                    generating_process=run_cache.load_generating_process() or 0,
                    member_ids=run_cache.load_member_ids() or (),
                    archived_at=self.clock(),
                    code_version=self._code_version,
                    load=lambda variable, member, step: run_cache.load(
                        variable=variable, member=member, step_minutes=step
                    ),
                )
            )
        except Exception as error:
            logger.warning(
                "commit of %s %s failed", product.name, init_time.isoformat(), exc_info=True
            )
            ledger.report(
                product=product.name,
                init_time=init_time,
                fault="commit_failed",
                detail=repr(error),
            )
            return False
        run_cache.delete()
        if status != STATUS_COMPLETE:
            fault: FaultType = "partial" if status == STATUS_PARTIAL else "missing"
            self.reporter.fault(
                product=product.name,
                init_time=init_time,
                fault=fault,
                detail=f"{received} of {n_files} files arrived by the deadline",
            )
        return True

    @staticmethod
    def _log_run(
        product: Product, init_time: datetime, expected: int, received: int, state: str
    ) -> None:
        """Log the one line each examined run gets per cycle."""
        logger.info(
            "product=%s init_time=%s expected=%d received=%d state=%s",
            product.name,
            init_time.isoformat(),
            expected,
            received,
            state,
        )

    def _check_free_space(self, *, product: Product, ledger: _FaultLedger) -> bool:
        """Warn when the cache disk is nearly full. Returns whether it is not."""
        self.config.cache_dir.mkdir(parents=True, exist_ok=True)
        free = shutil.disk_usage(self.config.cache_dir).free
        if free >= self.config.min_free_bytes:
            return True
        ledger.report(
            product=product.name,
            init_time=None,
            fault="disk_low",
            detail=f"{free / 1024**3:.1f} GiB free under {self.config.cache_dir}",
        )
        return False

    def _ensure_grid(
        self,
        *,
        source: Source,
        product: Product,
        store: ProductStore,
        run_cache: RunCache,
        ledger: _FaultLedger,
        init_time: datetime,
        past_deadline: bool,
    ) -> tuple[RunGrid | None, str | None]:
        """Get the run's cropped grid: cached, fetched, or (past the deadline) the stored one.

        A run fetches the full static set only when the archive holds no grid yet. Otherwise it
        fetches just the cell coordinates, to compare them with the archive's. A run whose grid
        differs from the archive's keeps its own grid, so that its files are still fetched.

        Args:
            source: The provider's files.
            product: The product being recorded.
            store: The product's repository.
            run_cache: The run's local cache.
            ledger: Where persistent faults are reported.
            init_time: The run's initialisation time.
            past_deadline: Whether the time has passed after which a run with no file is missing.

        Returns:
            The grid, or `None` if the files that give it are not published yet, and a description
            of how the grid differs from the archive's, or `None` if it does not.
        """
        stored = store.stored_grid()
        wanted = {"clat", "clon"} if stored is not None else None
        grid = run_cache.load_grid()
        if grid is None or (
            wanted is None and not source.static_names(product) <= grid.statics.keys()
        ):
            fetched = source.fetch_grid(
                product=product, init_time=init_time, only=wanted, previous=grid
            )
            if not isinstance(fetched, NotYet):
                grid = fetched
                run_cache.save_grid(grid)
        if grid is not None:
            return grid, store.grid_mismatch(grid)
        if stored is not None and past_deadline:
            ledger.report(
                product=product.name,
                init_time=init_time,
                fault="grid_unchecked",
                detail="the run's cell coordinates never arrived, so the stored grid was used",
            )
            return (
                RunGrid(
                    n_points=stored.n_points,
                    cell_index=stored.cell_index,
                    statics={"clat": stored.clat, "clon": stored.clon},
                    shape=stored.shape,
                ),
                None,
            )
        return None, None

    def _fetch_run(
        self,
        *,
        source: Source,
        files: list[ExpectedFile],
        grid: RunGrid,
        run_cache: RunCache,
        exhaustive: bool,
        past_deadline: bool,
        stop: Callable[[], bool] | None,
    ) -> tuple[int, str | None]:
        """Fetch a run's files, repeating the pass at the deadline if a pass saw a transient error.

        Args:
            source: The provider's files.
            files: The files the run should contain.
            grid: The run's cropped grid.
            run_cache: The run's local cache.
            exhaustive: Whether to try every missing file rather than stopping a sequence at its
                first file that is not yet published.
            past_deadline: Whether the run's deadline has passed.
            stop: Says when to give up on the rest of the run, or `None`.

        Returns:
            How many files were downloaded, and how the grid differs from the archive's, if it does.
        """
        n_fetched = 0
        grid_changed: str | None = None
        for _ in range(1 + (DEADLINE_RETRY_PASSES if past_deadline else 0)):
            result = self._fetch_pass(
                source=source,
                files=files,
                grid=grid,
                run_cache=run_cache,
                exhaustive=exhaustive,
                workers=self.config.backfill_workers if stop else self.config.workers,
                stop=stop,
            )
            n_fetched += result.n_fetched
            grid_changed = grid_changed or result.grid_changed
            if (
                result.generating_process is not None
                and run_cache.load_generating_process() is None
            ):
                run_cache.save_generating_process(result.generating_process)
            if result.member_ids and run_cache.load_member_ids() is None:
                run_cache.save_member_ids(result.member_ids)
            received = run_cache.count_received(files, n_cells=len(grid.cell_index))
            if received == len(files) or not result.transient:
                break
        return n_fetched, grid_changed

    def _fetch_pass(
        self,
        *,
        source: Source,
        files: list[ExpectedFile],
        grid: RunGrid,
        run_cache: RunCache,
        exhaustive: bool,
        workers: int,
        stop: Callable[[], bool] | None,
    ) -> _FetchPassResult:
        """Make one pass over the run's files, caching each one that has arrived.

        Files appear in step order, so unless `exhaustive` a sequence (as the source defines it)
        stops at its first file that is not yet published.
        """
        sequences: dict[tuple[str, ...], list[ExpectedFile]] = {}
        for file in files:
            sequences.setdefault(source.sequence_key(file), []).append(file)
        n_cells = len(grid.cell_index)

        def fetch_sequence(sequence: list[ExpectedFile]) -> _FetchPassResult:
            partial = _FetchPassResult()
            for file in sequence:
                if stop is not None and stop():
                    break
                if run_cache.has(file, n_cells=n_cells):
                    continue
                outcome = source.fetch(file, grid)
                if isinstance(outcome, NotYet):
                    if outcome.reason != "404":
                        partial.transient = True
                        logger.warning("%s is not usable yet: %s", file.url, outcome.reason)
                    if exhaustive:
                        continue
                    break
                if outcome.n_points != grid.n_points:
                    partial.grid_changed = (
                        f"a file has {outcome.n_points} cells, the archive's grid has "
                        f"{grid.n_points}"
                    )
                    continue
                run_cache.save(file, outcome.values)
                partial.n_fetched += 1
                partial.generating_process = outcome.generating_process
                partial.member_ids = outcome.member_ids
            return partial

        result = _FetchPassResult()
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for partial in pool.map(fetch_sequence, sequences.values()):
                result.grid_changed = result.grid_changed or partial.grid_changed
                result.n_fetched += partial.n_fetched
                result.transient |= partial.transient
                if partial.generating_process is not None:
                    result.generating_process = partial.generating_process
                result.member_ids = partial.member_ids or result.member_ids
        return result
