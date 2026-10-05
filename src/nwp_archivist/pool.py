"""Fetching a run's files by sequence, in threads or in worker processes.

A worker process fetches, decodes, and crops files and writes the cropped values to the run's local
cache. It never opens the repository, the fault ledger, or Sentry: the main process alone
commits and reports. A worker turns any exception into a transient "not yet" result, so one bad
file leaves its run waiting and never stops the others.
"""

import logging
import multiprocessing
import time
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from nwp_archivist.cache import RunCache, RunGrid
from nwp_archivist.products import ExpectedFile
from nwp_archivist.source import NotYet, Source

logger = logging.getLogger(__name__)


@dataclass
class FetchPassResult:
    """What one fetch pass over a run's files found."""

    generating_process: int | None = None
    grid_changed: str | None = None
    n_fetched: int = 0
    transient: bool = False
    member_ids: tuple[int, ...] = ()
    crashed: str | None = None


class FetchCrashedError(RuntimeError):
    """A worker process failed or the pool timed out, so the run's files are in an unknown state.

    This is our own fault and not the provider's, so a run is never committed after it.
    """


def fetch_sequence(
    *,
    source: Source,
    sequence: list[ExpectedFile],
    grid: RunGrid,
    run_cache: RunCache,
    exhaustive: bool,
    stop: Callable[[], bool] | None,
) -> FetchPassResult:
    """Fetch the files of one sequence in order, caching each one that has arrived.

    Unless `exhaustive`, the sequence stops at its first file that is not yet published.
    """
    result = FetchPassResult()
    n_cells = len(grid.cell_index)
    for file in sequence:
        if stop is not None and stop():
            break
        if run_cache.has(file, n_cells=n_cells):
            continue
        outcome = source.fetch(file, grid)
        if isinstance(outcome, NotYet):
            if outcome.reason != "404":
                result.transient = True
                logger.warning("%s is not usable yet: %s", file.url, outcome.reason)
            if exhaustive:
                continue
            break
        if outcome.n_points != grid.n_points:
            result.grid_changed = (
                f"a file has {outcome.n_points} cells, the archive's grid has {grid.n_points}"
            )
            continue
        run_cache.save(file, outcome.values)
        result.n_fetched += 1
        result.generating_process = outcome.generating_process
        result.member_ids = outcome.member_ids
    return result


_worker_source: Source | None = None


def _init_worker(factory: Callable[[], Source]) -> None:
    """Build this worker process's own source, with its own HTTP client and file handles."""
    global _worker_source  # noqa: PLW0603
    _worker_source = factory()


def _run_in_worker(
    sequence: list[ExpectedFile],
    grid: RunGrid,
    run_directory: Path,
    exhaustive: bool,
    stop_at: float | None,
) -> FetchPassResult:
    """Fetch one sequence in a worker process. `stop_at` is a `time.monotonic()` reading."""
    try:
        if _worker_source is None:
            message = "the worker has no source"
            raise RuntimeError(message)  # noqa: TRY301
        return fetch_sequence(
            source=_worker_source,
            sequence=sequence,
            grid=grid,
            run_cache=RunCache(directory=run_directory),
            exhaustive=exhaustive,
            stop=None if stop_at is None else (lambda: time.monotonic() >= stop_at),
        )
    except Exception as error:
        logger.warning("a fetch worker failed on %s", sequence[0].url, exc_info=True)
        return FetchPassResult(crashed=f"a fetch worker failed: {error!r}")


class FetchPool:
    """A pool of worker processes that fetch sequences of files into the local cache."""

    def __init__(self, *, processes: int, source_factory: Callable[[], Source]) -> None:
        """Start no processes yet; they start on first use.

        Args:
            processes: The number of worker processes.
            source_factory: A picklable, module-level function that builds a worker's source.
        """
        self._processes = processes
        self._source_factory = source_factory
        self._executor: ProcessPoolExecutor | None = None

    def _pool(self) -> ProcessPoolExecutor:
        if self._executor is None:
            # Spawn, so that no worker inherits the parent's HTTP connections or HDF5 state.
            self._executor = ProcessPoolExecutor(
                max_workers=self._processes,
                mp_context=multiprocessing.get_context("spawn"),
                initializer=_init_worker,
                initargs=(self._source_factory,),
            )
        return self._executor

    def run(
        self,
        *,
        sequences: list[list[ExpectedFile]],
        grid: RunGrid,
        run_cache: RunCache,
        exhaustive: bool,
        stop_at: float | None,
        timeout_seconds: float,
    ) -> list[FetchPassResult]:
        """Fetch every sequence, returning one result each.

        A sequence whose worker failed, or that had not finished after `timeout_seconds` for the
        whole pass, gets a result with `crashed` set. On such a failure the workers are terminated
        and waited for, so that none is left writing, and the next pass starts a fresh pool.
        """
        give_up = time.monotonic() + timeout_seconds
        futures = [
            self._pool().submit(
                _run_in_worker, sequence, grid, run_cache.directory, exhaustive, stop_at
            )
            for sequence in sequences
        ]
        results: list[FetchPassResult] = []
        for future in futures:
            try:
                results.append(future.result(timeout=max(give_up - time.monotonic(), 0.0)))
            except Exception as error:
                logger.warning("the fetch pool failed", exc_info=True)
                self.close()
                results.append(FetchPassResult(crashed=f"the fetch pool failed: {error!r}"))
        return results

    def close(self) -> None:
        """Terminate the worker processes and wait for them. A later `run` starts new ones."""
        executor, self._executor = self._executor, None
        if executor is None:
            return
        workers = list(getattr(executor, "_processes", {}).values())
        for worker in workers:
            worker.terminate()
        executor.shutdown(wait=True, cancel_futures=True)
        for worker in workers:
            worker.join()
