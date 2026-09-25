"""The local cache of decoded, cropped files, which is the recorder's checkpoint.

A file counts as received once its cropped values sit in the cache. Cropped values are far smaller
than the raw GRIB files, so the disk holds days of runs. The repository's status arrays, not this
cache, are the record of what is committed: the "done" markers here only save a repository read.
"""

import io
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from nwp_archivist.products import ExpectedFile

_GRID_FILE = "grid.npz"
_META_FILE = "meta.json"
_TIMESTAMP_FORMAT = "%Y%m%dT%H%M"


def _write_atomically(path: Path, data: bytes) -> None:
    """Write bytes so that a crash leaves either no file or the whole file, never a part."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as handle:
        handle.write(data)
    temporary.replace(path)


def _array_bytes(values: np.ndarray) -> bytes:
    """Serialise an array in `.npy` format."""
    buffer = io.BytesIO()
    np.save(buffer, values, allow_pickle=False)
    return buffer.getvalue()


@dataclass(frozen=True)
class RunGrid:
    """The cropped grid of one run.

    Attributes:
        n_points: The number of cells on the provider's full native grid.
        cell_index: The indices of the kept cells within the full grid.
        statics: The cropped static fields (`clat`, `clon`, and, when fetched, `hsurf`,
            `fr_land` and the half-level heights), keyed by their archive name.
    """

    n_points: int
    cell_index: np.ndarray
    statics: dict[str, np.ndarray]


class RunCache:
    """The cached files of one run of one product."""

    def __init__(self, *, directory: Path) -> None:
        """Point the cache at the run's directory, which need not exist yet."""
        self.directory = directory

    def _file_path(self, *, variable: str, member: int | None, step_minutes: int) -> Path:
        return self.directory / variable / f"{member or 0:02d}_{step_minutes:04d}.npy"

    def has(self, file: ExpectedFile) -> bool:
        """Whether the file's cropped values are cached."""
        return self._file_path(
            variable=file.field.variable,
            member=file.member,
            step_minutes=file.step_minutes,
        ).exists()

    def save(self, file: ExpectedFile, values: np.ndarray) -> None:
        """Cache the cropped values of one file."""
        path = self._file_path(
            variable=file.field.variable,
            member=file.member,
            step_minutes=file.step_minutes,
        )
        _write_atomically(path, _array_bytes(values))

    def load(self, *, variable: str, member: int | None, step_minutes: int) -> np.ndarray | None:
        """Read the cropped values of one file, or `None` if it was never received."""
        path = self._file_path(variable=variable, member=member, step_minutes=step_minutes)
        if not path.exists():
            return None
        return np.load(path, allow_pickle=False)

    def count_received(self, files: list[ExpectedFile]) -> int:
        """Count how many of `files` are cached."""
        return sum(1 for file in files if self.has(file))

    def save_grid(self, grid: RunGrid) -> None:
        """Cache the run's cropped grid."""
        buffer = io.BytesIO()
        contents: dict[str, np.ndarray] = {
            "n_points": np.asarray(grid.n_points, dtype=np.int64),
            "cell_index": grid.cell_index,
        }
        contents.update({f"static_{name}": values for name, values in grid.statics.items()})
        np.savez(buffer, **contents)  # ty: ignore[invalid-argument-type]
        _write_atomically(self.directory / _GRID_FILE, buffer.getvalue())

    def load_grid(self) -> RunGrid | None:
        """Read the run's cropped grid, or `None` if it was never cached."""
        path = self.directory / _GRID_FILE
        if not path.exists():
            return None
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
            )

    def save_generating_process(self, generating_process: int) -> None:
        """Remember the generating process identifier of the run's first decoded file."""
        _write_atomically(
            self.directory / _META_FILE,
            json.dumps({"generating_process": generating_process}).encode(),
        )

    def load_generating_process(self) -> int | None:
        """The remembered generating process identifier, or `None`."""
        path = self.directory / _META_FILE
        if not path.exists():
            return None
        return int(json.loads(path.read_text())["generating_process"])

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
