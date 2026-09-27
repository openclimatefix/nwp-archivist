"""Reading MOGREPS-G files from the Met Office's public S3 bucket, by byte range.

MOGREPS-G publishes one file per variable per lead time per run, holding all 18 members on the
full global 20 km regular latitude-longitude grid, as an HDF5 (NetCDF4) dataset chunked into 128
by 128 cell blocks (and, for the two height-level wind fields, one level per chunk too) and
compressed with deflate. Cropping the GB box therefore still means fetching only a handful of
chunks by byte range rather than the whole file: the archive keeps a 67 by 48 cell rectangle out of
960 by 1280, so about 4 of a level's 88 chunks are ever downloaded.

Everything except how a run's grid is chosen is identical to MOGREPS-UK's `MogrepsSource`, so this
module subclasses it and overrides only `fetch_grid`: MOGREPS-G's grid is a plain regular
latitude-longitude grid, not a Lambert azimuthal equal-area projection, so cropping it is a
coordinate threshold rather than an inverse projection.

As with MOGREPS-UK, a file that is absent, truncated, unreadable, or not the file its address
promised is "not yet". Nothing here raises for the provider misbehaving.
"""

from datetime import datetime
from typing import Final

import httpx
import numpy as np

from nwp_archivist.cache import RunGrid
from nwp_archivist.dwd import Fetcher
from nwp_archivist.mogreps import REQUEST_TIMEOUT_SECONDS, MogrepsSource
from nwp_archivist.products import BOX_LAT_MAX, BOX_LAT_MIN, BOX_LON_MAX, BOX_LON_MIN, Product
from nwp_archivist.source import NotYet

MOGREPS_G_BASE_URL: Final[str] = (
    "https://met-office-global-ensemble-model-data.s3.eu-west-2.amazonaws.com/global-ensemble"
)


def box_rectangle(*, latitude: np.ndarray, longitude: np.ndarray) -> RunGrid:
    """Choose the rectangle of grid cells that contains the crop box, on a regular lat/lon grid.

    Args:
        latitude: The cell-centre latitudes in degrees, one per row.
        longitude: The cell-centre longitudes in degrees, -180 to 180, one per column.

    Returns:
        The grid, whose kept cells are the smallest rectangle of rows and columns containing every
        cell whose centre lies inside the box (edges included), flattened row by row.
    """
    row_hits = np.flatnonzero((latitude >= BOX_LAT_MIN) & (latitude <= BOX_LAT_MAX))
    column_hits = np.flatnonzero((longitude >= BOX_LON_MIN) & (longitude <= BOX_LON_MAX))
    first_row, last_row = int(row_hits.min()), int(row_hits.max())
    first_column, last_column = int(column_hits.min()), int(column_hits.max())
    row_range = np.arange(first_row, last_row + 1)
    column_range = np.arange(first_column, last_column + 1)
    cell_index = np.add.outer(row_range * len(longitude), column_range).ravel().astype(np.int32)
    lat_grid, lon_grid = np.meshgrid(
        latitude[first_row : last_row + 1],
        longitude[first_column : last_column + 1],
        indexing="ij",
    )
    return RunGrid(
        n_points=len(latitude) * len(longitude),
        cell_index=cell_index,
        statics={
            "clat": lat_grid.ravel().astype(np.float32),
            "clon": lon_grid.ravel().astype(np.float32),
        },
        shape=(len(row_range), len(column_range)),
    )


class MogrepsGSource(MogrepsSource):
    """The Met Office's MOGREPS-G bucket: 18 members on one global regular lat/lon grid per file.

    Every method except `fetch_grid` is inherited unchanged from `MogrepsSource`: opening a file by
    byte range, checking it against its address, locating the 100 m height level, downloading the
    chunks under the crop rectangle, and splitting out one member, all work the same way regardless
    of the grid's projection.
    """

    def __init__(self, *, fetcher: Fetcher, base_url: str = MOGREPS_G_BASE_URL) -> None:
        """Build a source. See `MogrepsSource.__init__`; only the default bucket address differs."""
        super().__init__(fetcher=fetcher, base_url=base_url)

    def fetch_grid(
        self,
        *,
        product: Product,
        init_time: datetime,
        only: set[str] | None,
        previous: RunGrid | None,
    ) -> RunGrid | NotYet:
        """Read the grid coordinates from a lead-time-0 file of the run and choose the crop."""
        files = self.expected_files(product, init_time)
        reference = next(file for file in files if file.step_minutes == 0)
        try:
            opened = self._open(reference.url)
            if isinstance(opened, NotYet):
                return opened
            file = opened.file
            grid = box_rectangle(latitude=file["latitude"][:], longitude=file["longitude"][:])
            return NotYet(opened.reader.failure) if opened.reader.failure else grid
        except (OSError, KeyError, ValueError, RuntimeError, ArithmeticError) as error:
            return self._unreadable(error)


def make_mogreps_g_source() -> MogrepsGSource:
    """Build a source with its own HTTP client, for a worker process."""
    return MogrepsGSource(fetcher=Fetcher(client=httpx.Client(timeout=REQUEST_TIMEOUT_SECONDS)))
