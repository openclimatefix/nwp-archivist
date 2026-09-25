"""The recording cycle: work out which runs should exist, fetch their files, and commit them.

Nothing here raises because a provider file is absent or late. A run is retried every cycle, and
once its deadline passes it is committed with whatever arrived, marked `partial`, or marked
`missing` if nothing arrived. Each run sits in its own `try` block, so one failing run never stops
the others.
"""

import logging
import shutil
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Final

import numpy as np

from nwp_archivist.cache import ProductCache, RunCache, RunGrid
from nwp_archivist.dwd import Decoded, Expectation, Fetcher, NotYet
from nwp_archivist.products import (
    DWD_BASE_URL,
    ExpectedFile,
    Product,
    expected_files,
    run_init_times,
    static_urls,
)
from nwp_archivist.reporting import FaultType, Reporter
from nwp_archivist.store import (
    STATUS_COMPLETE,
    STATUS_MISSING,
    STATUS_NAMES,
    STATUS_PARTIAL,
    ProductStore,
    RunToCommit,
    StoreLocation,
    box_cell_index,
    slot_for,
)

logger = logging.getLogger(__name__)

DEFAULT_MIN_FREE_BYTES: Final[int] = 10 * 1024**3


class GridChangedError(Exception):
    """A run's grid, member count, or step list differs from the archive's."""


def code_version() -> str:
    """The installed version of this package, or `unknown` when it is not installed."""
    try:
        return version("nwp-archivist")
    except PackageNotFoundError:
        return "unknown"


@dataclass(frozen=True)
class RecorderConfig:
    """The settings of a recording cycle.

    Attributes:
        store: Where the product repositories live.
        cache_dir: Where cropped files are cached until their run is committed.
        min_free_bytes: The free disk space under which each run reports a `disk_low` fault.
        lookback_hours: How far back from now a cycle looks for runs that are not yet archived.
        workers: The number of files downloaded in parallel.
        base_url: The root of the provider's directory layout.
    """

    store: StoreLocation
    cache_dir: Path
    min_free_bytes: int = DEFAULT_MIN_FREE_BYTES
    lookback_hours: float = 36.0
    workers: int = 8
    base_url: str = DWD_BASE_URL


@dataclass
class _FetchPassResult:
    """What one fetch pass over a run's files found."""

    generating_process: int | None = None
    grid_changed: str | None = None


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _slot_archived(product: Product, statuses: np.ndarray, init_time: datetime) -> bool:
    """Whether an already-read status array holds this run."""
    slot = slot_for(product, init_time)
    return slot < len(statuses) and statuses[slot] != 0


class Recorder:
    """Runs recording cycles for a set of products."""

    def __init__(
        self,
        *,
        config: RecorderConfig,
        fetcher: Fetcher,
        reporter: Reporter,
        clock: Callable[[], datetime] = _utc_now,
    ) -> None:
        """Build a recorder.

        Args:
            config: The recorder's settings.
            fetcher: The file downloader.
            reporter: Where faults and the cycle check-in go.
            clock: Returns the current time, replaceable so that tests control time.
        """
        self.config = config
        self.fetcher = fetcher
        self.reporter = reporter
        self.clock = clock
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
        halted = cache.halted()
        if halted is not None:
            logger.info("product=%s state=halted reason=%s", product.name, halted)
            return False
        try:
            store = ProductStore.open(location=self.config.store, product=product)
            mismatch = store.layout_mismatch()
            archived = store.statuses()
        except Exception as error:
            logger.exception("cannot open the %s repository", product.name)
            self.reporter.fault(
                product=product.name,
                init_time=None,
                fault="commit_failed",
                detail=f"cannot open the repository: {error!r}",
            )
            return False
        if mismatch is not None:
            self._halt(cache=cache, product=product, reason=mismatch)
            return False
        now = self.clock()
        clean = True
        runs = run_init_times(
            product,
            first=now - timedelta(hours=self.config.lookback_hours),
            last=now - timedelta(hours=product.start_delay_hours),
        )
        for init_time in runs:
            if cache.is_done(init_time) or _slot_archived(store.product, archived, init_time):
                continue
            if cache.halted() is not None:
                return False
            try:
                clean &= self._process_run(
                    product=product,
                    store=store,
                    cache=cache,
                    init_time=init_time,
                    now=now,
                )
            except GridChangedError as error:
                self._halt(cache=cache, product=product, reason=str(error), init_time=init_time)
                return False
            except Exception as error:
                logger.exception("run %s %s failed", product.name, init_time.isoformat())
                self.reporter.fault(
                    product=product.name,
                    init_time=init_time,
                    fault="run_error",
                    detail=repr(error),
                )
                clean = False
        return clean

    def _halt(
        self,
        *,
        cache: ProductCache,
        product: Product,
        reason: str,
        init_time: datetime | None = None,
    ) -> None:
        """Stop commits for a product and report it once."""
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
        init_time: datetime,
        now: datetime,
    ) -> bool:
        """Fetch what is missing of one run and commit it if it is complete or past its deadline.

        Args:
            product: The product being recorded.
            store: The product's repository.
            cache: The product's local cache.
            init_time: The run's initialisation time.
            now: The time this cycle started.

        Returns:
            Whether the run was handled without a fault.
        """
        self._check_free_space(product=product, init_time=init_time)
        files = expected_files(product, init_time, base_url=self.config.base_url)
        run_cache = cache.run(init_time)
        past_deadline = now >= product.deadline(init_time)
        grid = self._ensure_grid(
            product=product,
            store=store,
            run_cache=run_cache,
            init_time=init_time,
            past_deadline=past_deadline,
        )
        if grid is not None:
            result = self._fetch_pass(
                files=files, grid=grid, run_cache=run_cache, exhaustive=past_deadline
            )
            if result.grid_changed is not None:
                raise GridChangedError(result.grid_changed)
            if (
                result.generating_process is not None
                and run_cache.load_generating_process() is None
            ):
                run_cache.save_generating_process(result.generating_process)
        received = run_cache.count_received(files)
        if received == len(files):
            status = STATUS_COMPLETE
        elif past_deadline:
            status = STATUS_PARTIAL if received > 0 else STATUS_MISSING
        else:
            self._log_run(product, init_time, len(files), received, "waiting")
            return True
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
            return False
        return self._commit(
            product=product,
            store=store,
            cache=cache,
            run_cache=run_cache,
            init_time=init_time,
            grid=grid,
            status=status,
            n_files=len(files),
            received=received,
        )

    def _commit(
        self,
        *,
        product: Product,
        store: ProductStore,
        cache: ProductCache,
        run_cache: RunCache,
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
                    archived_at=self.clock(),
                    code_version=self._code_version,
                    load=lambda variable, member, step: run_cache.load(
                        variable=variable, member=member, step_minutes=step
                    ),
                )
            )
        except Exception as error:
            logger.exception("commit of %s %s failed", product.name, init_time.isoformat())
            self.reporter.fault(
                product=product.name,
                init_time=init_time,
                fault="commit_failed",
                detail=repr(error),
            )
            return False
        cache.mark_done(init_time)
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

    def _check_free_space(self, *, product: Product, init_time: datetime) -> None:
        """Warn when the cache disk is nearly full, so a full disk is not a surprise."""
        self.config.cache_dir.mkdir(parents=True, exist_ok=True)
        free = shutil.disk_usage(self.config.cache_dir).free
        if free < self.config.min_free_bytes:
            self.reporter.fault(
                product=product.name,
                init_time=init_time,
                fault="disk_low",
                detail=f"{free / 1024**3:.1f} GiB free under {self.config.cache_dir}",
            )

    def _ensure_grid(
        self,
        *,
        product: Product,
        store: ProductStore,
        run_cache: RunCache,
        init_time: datetime,
        past_deadline: bool,
    ) -> RunGrid | None:
        """Get the run's cropped grid: cached, fetched, or (past the deadline) the stored one.

        A run fetches the full static set only when the archive holds no grid yet. Otherwise it
        fetches just the cell coordinates, to compare them with the archive's.

        Args:
            product: The product being recorded.
            store: The product's repository.
            run_cache: The run's local cache.
            init_time: The run's initialisation time.
            past_deadline: Whether the run's deadline has passed.

        Raises:
            GridChangedError: If the run's grid differs from the archive's.

        Returns:
            The grid, or `None` if the files that give it are not published yet.
        """
        stored = store.stored_grid()
        wanted = {"clat", "clon"} if stored is not None else None
        grid = run_cache.load_grid()
        if grid is None or (wanted is None and not self._has_all_statics(product, grid)):
            grid = self._fetch_grid(
                product=product, init_time=init_time, only=wanted, previous=grid
            )
            if grid is not None:
                run_cache.save_grid(grid)
        if grid is not None:
            mismatch = store.grid_mismatch(grid)
            if mismatch is not None:
                raise GridChangedError(mismatch)
            return grid
        if stored is not None and past_deadline:
            self.reporter.fault(
                product=product.name,
                init_time=init_time,
                fault="grid_unchecked",
                detail="the run's cell coordinates never arrived, so the stored grid was used",
            )
            return RunGrid(
                n_points=stored.n_points,
                cell_index=stored.cell_index,
                statics={"clat": stored.clat, "clon": stored.clon},
            )
        return None

    @staticmethod
    def _has_all_statics(product: Product, grid: RunGrid) -> bool:
        needed = {"clat", "clon", "hsurf", "fr_land"} | {f"hhl_L{n}" for n in product.hhl_levels}
        return needed <= grid.statics.keys()

    def _fetch_grid(
        self,
        *,
        product: Product,
        init_time: datetime,
        only: set[str] | None,
        previous: RunGrid | None,
    ) -> RunGrid | None:
        """Download the static fields and crop them, or return `None` if any is not yet there."""
        urls = static_urls(product, init_time, base_url=self.config.base_url)
        wanted = {name: url for name, url in urls.items() if only is None or name in only}
        decoded: dict[str, Decoded] = {}
        for name, url in wanted.items():
            level = int(name.removeprefix("hhl_L")) if name.startswith("hhl_L") else None
            outcome = self.fetcher.fetch(url, expect=Expectation(level=level))
            if isinstance(outcome, NotYet):
                logger.info("static %s of %s: %s", name, init_time.isoformat(), outcome.reason)
                return None
            decoded[name] = outcome
        clat = decoded["clat"].values
        clon = decoded["clon"].values
        cell_index = box_cell_index(clat=clat, clon=clon)
        statics = {name: values.values[cell_index] for name, values in decoded.items()}
        if previous is not None:
            statics = {**previous.statics, **statics}
        return RunGrid(n_points=len(clat), cell_index=cell_index, statics=statics)

    def _fetch_pass(
        self,
        *,
        files: list[ExpectedFile],
        grid: RunGrid,
        run_cache: RunCache,
        exhaustive: bool,
    ) -> _FetchPassResult:
        """Make one pass over the run's files, caching each one that has arrived.

        Files appear in step order, so unless `exhaustive` a sequence (one variable of one member)
        stops at its first file that is not yet published. After the deadline every file is
        tried.
        """
        sequences: dict[tuple[str, int | None], list[ExpectedFile]] = {}
        for file in files:
            sequences.setdefault((file.field.variable, file.member), []).append(file)
        result = _FetchPassResult()

        def fetch_sequence(sequence: list[ExpectedFile]) -> _FetchPassResult:
            partial = _FetchPassResult()
            for file in sequence:
                if run_cache.has(file):
                    continue
                expect = Expectation(
                    short_name=file.field.short_name,
                    step_minutes=file.step_minutes,
                    member=file.member,
                    level=file.field.level,
                )
                outcome = self.fetcher.fetch(file.url, expect=expect)
                if isinstance(outcome, NotYet):
                    if outcome.reason != "404":
                        logger.warning("%s is not usable yet: %s", file.url, outcome.reason)
                    if exhaustive:
                        continue
                    break
                if len(outcome.values) != grid.n_points:
                    partial.grid_changed = (
                        f"a file has {len(outcome.values)} cells, the archive's grid has "
                        f"{grid.n_points}"
                    )
                    return partial
                run_cache.save(file, outcome.values[grid.cell_index])
                partial.generating_process = outcome.generating_process
            return partial

        with ThreadPoolExecutor(max_workers=self.config.workers) as pool:
            for partial in pool.map(fetch_sequence, sequences.values()):
                result.grid_changed = result.grid_changed or partial.grid_changed
                if partial.generating_process is not None:
                    result.generating_process = partial.generating_process
        return result
