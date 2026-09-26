# nwp-archivist

An always-on recorder of ensemble numerical weather prediction (NWP) products that their providers
delete within 24 hours to 33 days. It records the DWD ICON-EU-EPS, ICON-D2-EPS, ICON-D2, and
ICON-ART-EU products, cropped to Great Britain and its surrounding seas (49.0-61.5 N, 10.0 W-3.5 E),
into one [Icechunk](https://icechunk.io) repository per product. It also records the Met Office
MOGREPS-UK ensemble (`mogreps-uk`, 3 members, hourly runs to 126 hours) as a separate product
because of its licence. The design, its reviews, and the plan are in
[openclimatefix/nged-substation-forecast#926](https://github.com/openclimatefix/nged-substation-forecast/issues/926).

## Licences

The code is MIT licensed (see `LICENSE`). The archived data keeps the licence of its provider: DWD
data is CC BY 4.0 (attribute the Deutscher Wetterdienst), and Met Office MOGREPS-UK data is CC
BY-SA 4.0.

## How to run

`archive-record` runs one recording cycle and exits. A systemd timer runs it every 15 minutes.

```bash
uv sync
export ARCHIVE_STORE_ROOT=/mnt/data/nwp-archive   # or an s3:// prefix
uv run archive-record --cache-dir /mnt/data/nwp-archive-cache
```

- `--store-root` (or `ARCHIVE_STORE_ROOT`) is a local directory or an `s3://bucket/prefix`
  address. Each product is the repository `<root>/<product name>`.
- `--cache-dir` holds cropped files until their run is committed (default
  `/mnt/data/nwp-archive-cache`).
- `SENTRY_DSN`, when set, sends faults and a cron check-in to Sentry. Without it, faults are logged
  at WARNING and the check-in does nothing.
- `deploy/nwp-archive.service` and `deploy/nwp-archive.timer` are systemd user units for a
  workstation. Run `loginctl enable-linger` so that they run while nobody is logged in.

Each cycle logs one line per examined run to standard output: the product, the init time, the
number of files expected and received, and the state (`waiting`, `complete`, `partial`, or
`missing`).

If a product's grid, member count, or step list differs from the archive's, the recorder stops
committing that product, reports one `grid_changed` fault, and creates `<cache-dir>/<product>/HALTED`.
Delete that file once a person has decided what to do.

## MOGREPS-UK

`archive-record --products mogreps-uk` records the Met Office's public bucket
`met-office-uk-ensemble-model-data`, which deletes each file about 30 days after it was written. The
`deploy/nwp-archive-mogreps.service` and `.timer` run it as a chain of cycles, each starting 1 minute
after the last exits, with a cache directory and a Sentry cron monitor (`nwp-archive-mogreps`) of
their own. The DWD service uses the monitor `nwp-archive-dwd`. Override either with `--monitor-slug`.

- **Fields**, all as delivered: total, direct, and diffuse downward shortwave at the surface, screen
  temperature, 10 m wind speed and direction, total cloud amount, and wind speed and direction at
  100 m. Shortwave has no lead time 0 and carries no `cell_methods` or time bounds, so it is taken
  to be instantaneous.
- **Grid:** the native 2 km grid, cut to the smallest rectangle of rows and columns that contains
  the crop box (707 rows by 494 columns), flattened row by row into the `cell` dimension. The
  `grid_shape` attribute holds the rows and columns.
- **Members** are numbered 1 to 3 in file order. The `realization` array holds the Met Office's
  number for each member, which changes from run to run.
- **Reading** downloads only the chunks that overlap the rectangle, by byte range. One measured run
  took 5.5 GB of downloads in 54,000 requests and 29 minutes with 8 threads, and became 1.06 GB of
  archive.
- **Backfill:** each cycle handles the runs from the last 6 hours first. It then spends at most
  `--backfill-minutes` (default 15; the service uses 10) on older runs still in the bucket, main runs
  (00, 06, 12 and 18 UTC) first and the other hours after them, each newest first, with fewer download threads, so the backfill goes on in slices. The bucket holds about 720
  runs and one run takes about 29 minutes, so the backfill takes weeks and the oldest runs may
  expire before it reaches them. A run with no file at all is recorded `missing` only 29 days after its
  initialisation time, and until then it is looked at again after a delay that doubles from 30
  minutes to 12 hours.

## Reading the archive

Each variable is one array with dimensions `(init_time, member, step, cell)`, or `(init_time,
step, cell)` for a deterministic product. Open a repository with `icechunk` and `xarray`:

```python
import icechunk
import xarray as xr

repo = icechunk.Repository.open(
    icechunk.local_filesystem_storage("/mnt/data/nwp-archive/icon-d2-eps")
)
session = repo.readonly_session("main")
dataset = xr.open_zarr(session.store, consolidated=False, decode_timedelta=True)
```

The `status` array says whether each `init_time` slot is `complete` (1), `partial` (2), or `missing`
(3); 0 means never archived. The `step` coordinate is in minutes, and a variable that lacks a
lead time has NaN at that position. Shortwave radiation (`ASWDIR_S`, `ASWDIFD_S`) is an average
since the start of the run, as delivered.

## Tests

```bash
uv run ruff check . && uv run ruff format --check . && uv run ty check && uv run pytest
uv run pytest --run-network -m network   # fetches three real files from DWD
```

## Contributing

Pull requests are limited to members of Open Climate Fix.
