# nwp-archivist

An always-on recorder of ensemble numerical weather prediction (NWP) products that their providers
delete within 24 hours to 33 days. It records the DWD ICON-EU-EPS, ICON-D2-EPS, ICON-D2, and
ICON-ART-EU products, cropped to Great Britain and its surrounding seas (49.0-61.5 N, 10.0 W-3.5 E),
into one [Icechunk](https://icechunk.io) repository per product. The Met Office MOGREPS-UK ensemble
is planned. The design, its reviews, and the plan are in
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
