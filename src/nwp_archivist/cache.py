"""The local cache of decoded, cropped files, which is the recorder's checkpoint.

A file counts as received once its cropped values sit in the cache. Cropped values are far smaller
than the raw GRIB files, so the disk holds days of runs. The repository's status arrays, not this
cache, are the record of what is committed: the "done" markers here only save a repository read.
"""

import io
import json
import os
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path

import numpy as np

from nwp_archivist.products import ExpectedFile

_GRID_FILE = "grid.npz"
_META_FILE = "meta.json"
_MEMBERS_FILE = "members.json"
_BACKOFF_FILE = "backoff.json"
_FAULTS_FILE = "active_faults.json"
_TIMESTAMP_FORMAT = "%Y%m%dT%H%M"


def _write_atomically(path: Path, data: bytes) -> None:
    """Write bytes so that a crash leaves either no file or the whole file, never a part."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # The process id keeps two writers of one path, such as a worker that outlived its pool, apart.
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        handle.write(data)
    temporary.replace(path)


_UNREADABLE = (OSError, ValueError, EOFError, zipfile.BadZipFile)


def _load_or_delete(path: Path) -> np.ndarray | None:
    """Read an `.npy` file, or delete it and return `None` if it is absent or unreadable."""
    try:
        return np.load(path, allow_pickle=False)
    except FileNotFoundError:
        return None
    except _UNREADABLE:
        path.unlink(missing_ok=True)
        return None


def _array_bytes(values: np.ndarray) -> bytes:
    """Serialise an array in `.npy` format."""
    buffer = io.BytesIO()
    np.save(buffer, values, allow_pickle=False)
    return buffer.getvalue()


@lru_cache(maxsize=8)
def _npy_size(n_cells: int) -> int:
    """The size in bytes of a cached file holding `n_cells` `float32` values."""
    return len(_array_bytes(np.zeros(n_cells, dtype=np.float32)))


@dataclass(frozen=True)
class RunGrid:
    """The cropped grid of one run.

    Attributes:
        n_points: The number of cells on the provider's full native grid.
        cell_index: The indices of the kept cells within the full grid.
        statics: The cropped static fields (`clat`, `clon`, and, when fetched, `hsurf`,
            `fr_land` and the half-level heights), keyed by their archive name.
        shape: The number of rows and columns of the kept cells, when they are a rectangle of a
            regular grid.
    """

    n_points: int
    cell_index: np.ndarray
    statics: dict[str, np.ndarray]
    shape: tuple[int, int] | None = None


class RunCache:
    """The cached files of one run of one product."""

    def __init__(self, *, directory: Path) -> None:
        """Point the cache at the run's directory, which need not exist yet."""
        self.directory = directory

    def _file_path(self, *, variable: str, member: int | None, step_minutes: int) -> Path:
        return self.directory / variable / f"{member or 0:02d}_{step_minutes:04d}.npy"

    def has(self, file: ExpectedFile, *, n_cells: int) -> bool:
        """Whether the file's cropped values are cached whole.

        A file of the wrong size (a write cut short by a crash, or an empty file left by a power
        loss) is deleted, so that the next fetch pass downloads it again.
        """
        path = self._file_path(
            variable=file.field.variable, member=file.member, step_minutes=file.step_minutes
        )
        try:
            size = path.stat().st_size
        except FileNotFoundError:
            return False
        if size != _npy_size(n_cells):
            path.unlink(missing_ok=True)
            return False
        return True

    def save(self, file: ExpectedFile, values: np.ndarray) -> None:
        """Cache the cropped values of one file."""
        path = self._file_path(
            variable=file.field.variable,
            member=file.member,
            step_minutes=file.step_minutes,
        )
        _write_atomically(path, _array_bytes(values))

    def load(self, *, variable: str, member: int | None, step_minutes: int) -> np.ndarray | None:
        """Read the cropped values of one file, or `None` if it is absent or unreadable.

        An unreadable file is deleted, so that the next fetch pass downloads it again.
        """
        path = self._file_path(variable=variable, member=member, step_minutes=step_minutes)
        return _load_or_delete(path)

    def count_received(self, files: list[ExpectedFile], *, n_cells: int) -> int:
        """Count how many of `files` are cached whole."""
        return sum(1 for file in files if self.has(file, n_cells=n_cells))

    def save_grid(self, grid: RunGrid) -> None:
        """Cache the run's cropped grid."""
        buffer = io.BytesIO()
        contents: dict[str, np.ndarray] = {
            "n_points": np.asarray(grid.n_points, dtype=np.int64),
            "cell_index": grid.cell_index,
        }
        if grid.shape is not None:
            contents["shape"] = np.asarray(grid.shape, dtype=np.int64)
        contents.update({f"static_{name}": values for name, values in grid.statics.items()})
        np.savez(buffer, **contents)  # ty: ignore[invalid-argument-type]
        _write_atomically(self.directory / _GRID_FILE, buffer.getvalue())

    def load_grid(self) -> RunGrid | None:
        """Read the run's cropped grid, or `None` if it is absent or unreadable.

        An unreadable file is deleted, so that the grid is fetched again.
        """
        path = self.directory / _GRID_FILE
        try:
            with np.load(path, allow_pickle=False) as stored:
                statics = {
                    key.removeprefix("static_"): stored[key]
                    for key in stored.files
                    if key.startswith("static_")
                }
                return RunGrid(
                    n_points=int(stored["n_points"]),
                    cell_index=stored["cell_index"],
                    statics=statics,
                    shape=(
                        (int(stored["shape"][0]), int(stored["shape"][1]))
                        if "shape" in stored.files
                        else None
                    ),
                )
        except FileNotFoundError:
            return None
        except _UNREADABLE:
            path.unlink(missing_ok=True)
            return None

    def save_generating_process(self, generating_process: int) -> None:
        """Remember the generating process identifier of the run's first decoded file."""
        _write_atomically(
            self.directory / _META_FILE,
            json.dumps({"generating_process": generating_process}).encode(),
        )

    def load_generating_process(self) -> int | None:
        """The remembered generating process identifier, or `None`."""
        path = self.directory / _META_FILE
        try:
            return int(json.loads(path.read_text())["generating_process"])
        except FileNotFoundError:
            return None
        except (*_UNREADABLE, KeyError):
            path.unlink(missing_ok=True)
            return None

    def save_member_ids(self, member_ids: tuple[int, ...]) -> None:
        """Remember the provider's numbers for the run's members, in member order."""
        _write_atomically(self.directory / _MEMBERS_FILE, json.dumps(member_ids).encode())

    def load_member_ids(self) -> tuple[int, ...] | None:
        """The remembered member numbers, or `None`."""
        path = self.directory / _MEMBERS_FILE
        try:
            return tuple(int(number) for number in json.loads(path.read_text()))
        except FileNotFoundError:
            return None
        except (*_UNREADABLE, TypeError):
            path.unlink(missing_ok=True)
            return None

    def save_backoff(self, *, attempts: int, next_try: datetime) -> None:
        """Remember that a pass found nothing of this run, and when to look again."""
        _write_atomically(
            self.directory / _BACKOFF_FILE,
            json.dumps({"attempts": attempts, "next_try": next_try.isoformat()}).encode(),
        )

    def load_backoff(self) -> tuple[int, datetime] | None:
        """The number of empty passes so far and when to look again, or `None`."""
        path = self.directory / _BACKOFF_FILE
        try:
            stored = json.loads(path.read_text())
            return int(stored["attempts"]), datetime.fromisoformat(stored["next_try"])
        except FileNotFoundError:
            return None
        except (*_UNREADABLE, KeyError, TypeError):
            path.unlink(missing_ok=True)
            return None

    def delete(self) -> None:
        """Delete the run's cached files, after its commit succeeded."""
        if not self.directory.exists():
            return
        for path in sorted(self.directory.rglob("*"), reverse=True):
            if path.is_dir():
                path.rmdir()
            else:
                path.unlink()
        self.directory.rmdir()


class ProductCache:
    """The local cache of one product: its runs, its "done" markers, and its halt marker."""

    def __init__(self, *, root: Path, product: str) -> None:
        """Point the cache at `root/product`."""
        self.directory = root / product

    def run(self, init_time: datetime) -> RunCache:
        """The cache of one run."""
        return RunCache(directory=self.directory / "runs" / init_time.strftime(_TIMESTAMP_FORMAT))

    def cached_runs(self) -> list[datetime]:
        """The init times of every run that still has a cache directory, oldest first."""
        runs_directory = self.directory / "runs"
        if not runs_directory.exists():
            return []
        times: list[datetime] = []
        for path in runs_directory.iterdir():
            try:
                times.append(datetime.strptime(path.name, _TIMESTAMP_FORMAT).replace(tzinfo=UTC))
            except ValueError:
                continue
        return sorted(times)

    def is_done(self, init_time: datetime) -> bool:
        """Whether the run was committed, according to this cache."""
        return (self.directory / "done" / init_time.strftime(_TIMESTAMP_FORMAT)).exists()

    def mark_done(self, init_time: datetime) -> None:
        """Record that the run was committed."""
        _write_atomically(self.directory / "done" / init_time.strftime(_TIMESTAMP_FORMAT), b"")

    @property
    def _halt_path(self) -> Path:
        return self.directory / "HALTED"

    def halted(self) -> str | None:
        """The reason commits for this product are stopped, or `None` if they are not."""
        if not self._halt_path.exists():
            return None
        return self._halt_path.read_text()

    def halt(self, reason: str) -> None:
        """Stop commits for this product until a person deletes the `HALTED` file."""
        _write_atomically(self._halt_path, reason.encode())

    def load_active_faults(self) -> set[str]:
        """The keys of the persistent faults reported and not yet cleared."""
        try:
            return set(json.loads((self.directory / _FAULTS_FILE).read_text()))
        except (FileNotFoundError, *_UNREADABLE):
            return set()

    def save_active_faults(self, keys: set[str]) -> None:
        """Remember which persistent faults are active, so the next cycle does not repeat them."""
        _write_atomically(self.directory / _FAULTS_FILE, json.dumps(sorted(keys)).encode())
