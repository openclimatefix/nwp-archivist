"""What the recorder needs from a provider, so that one recording cycle serves every provider."""

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

import numpy as np

from nwp_archivist.cache import RunGrid
from nwp_archivist.products import ExpectedFile, Product


@dataclass(frozen=True)
class NotYet:
    """The outcome of a fetch that produced no usable file.

    Attributes:
        reason: A short description of what was wrong, for the log.
    """

    reason: str


@dataclass(frozen=True)
class Cropped:
    """One field of one member, cropped to the archive's grid.

    Attributes:
        values: The cropped field as `float32`, with masked or missing cells set to NaN.
        n_points: The number of cells on the provider's full native grid, which the recorder
            compares with the archive's.
        generating_process: A number that changes when the provider changes its weather model,
            or 0 where the file carries none.
        member_ids: The provider's numbers for the file's members, in member order, where the
            provider numbers members differently in each run.
    """

    values: np.ndarray
    n_points: int
    generating_process: int
    member_ids: tuple[int, ...] = ()


class Source(Protocol):
    """One provider's files: which ones a run holds, how to read them, and how to crop them."""

    def expected_files(self, product: Product, init_time: datetime) -> list[ExpectedFile]:
        """Every file a run should contain, in the order a fetcher should try them."""
        ...

    def sequence_key(self, file: ExpectedFile) -> tuple[str, ...]:
        """Group files into sequences.

        One thread fetches a sequence in order and stops at its first file not yet published.
        """
        ...

    def static_names(self, product: Product) -> set[str]:
        """The names of the grid fields the archive stores once."""
        ...

    def fetch_grid(
        self,
        *,
        product: Product,
        init_time: datetime,
        only: set[str] | None,
        previous: RunGrid | None,
    ) -> RunGrid | NotYet:
        """Read the run's grid fields (only those in `only`, if given) and choose the cells to keep.

        `previous` holds the grid fields already fetched for this run, to be kept.
        """
        ...

    def fetch(self, file: ExpectedFile, grid: RunGrid) -> Cropped | NotYet:
        """Download, check, and crop one file. Nothing here raises for a provider's misbehaviour."""
        ...
