import fcntl
import logging
from pathlib import Path
from typing import ClassVar

import pytest

from nwp_archivist import cli
from nwp_archivist.products import PRODUCTS, Product
from nwp_archivist.recorder import RecorderConfig
from nwp_archivist.reporting import Reporter


class _StubRecorder:
    """Stands in for `Recorder`, keeping how the command built it."""

    instances: ClassVar[list[_StubRecorder]] = []

    def __init__(self, *, config: RecorderConfig, reporter: Reporter, **_: object) -> None:
        self.config = config
        self.products: list[Product] = []
        _StubRecorder.instances.append(self)

    def run_cycle(self, products: list[Product]) -> bool:
        self.products = products
        return True


@pytest.fixture(autouse=True)
def _stub_recorder(monkeypatch: pytest.MonkeyPatch) -> None:
    _StubRecorder.instances.clear()
    monkeypatch.setattr(cli, "Recorder", _StubRecorder)
    monkeypatch.delenv("SENTRY_DSN", raising=False)
    monkeypatch.delenv("ARCHIVE_STORE_ROOT", raising=False)


def test_without_a_store_root_the_command_exits_with_2() -> None:
    assert cli.main([]) == 2
    assert _StubRecorder.instances == []


def test_the_store_root_comes_from_the_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("ARCHIVE_STORE_ROOT", "s3://bucket/prefix")
    assert cli.main(["--cache-dir", str(tmp_path)]) == 0
    assert _StubRecorder.instances[0].config.store.root == "s3://bucket/prefix"


def test_the_flags_reach_the_recorder_and_the_dwd_products_run_by_default(tmp_path: Path) -> None:
    code = cli.main(
        ["--store-root", str(tmp_path / "s"), "--cache-dir", str(tmp_path / "c"), "--workers", "3"]
    )
    assert code == 0
    recorder = _StubRecorder.instances[0]
    assert recorder.config.cache_dir == tmp_path / "c"
    assert recorder.config.workers == 3
    assert {p.name for p in recorder.products} == {
        name for name, product in PRODUCTS.items() if product.source == "dwd"
    }
    assert "mogreps-uk" not in {p.name for p in recorder.products}


def test_the_products_flag_selects_products(tmp_path: Path) -> None:
    cli.main(
        ["--store-root", str(tmp_path), "--cache-dir", str(tmp_path / "c"), "--products", "icon-d2"]
    )
    assert [p.name for p in _StubRecorder.instances[0].products] == ["icon-d2"]


def test_mogreps_is_recorded_when_asked_for_and_gets_its_backfill_budget(tmp_path: Path) -> None:
    cli.main(
        [
            "--store-root",
            str(tmp_path),
            "--cache-dir",
            str(tmp_path / "c"),
            "--products",
            "mogreps-uk",
            "--backfill-minutes",
            "7",
        ]
    )
    recorder = _StubRecorder.instances[0]
    assert [p.name for p in recorder.products] == ["mogreps-uk"]
    assert recorder.config.backfill_seconds == 7 * 60


def test_the_default_cache_directory_is_on_the_data_disk() -> None:
    assert cli.build_parser().parse_args([]).cache_dir == "/mnt/data/nwp-archive-cache"


def test_a_second_process_exits_zero_when_another_holds_the_lock(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    (tmp_path / "c").mkdir()
    with (tmp_path / "c" / ".lock").open("w") as held:
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with caplog.at_level(logging.INFO):
            code = cli.main(
                ["--store-root", str(tmp_path / "s"), "--cache-dir", str(tmp_path / "c")]
            )
    assert code == 0
    assert _StubRecorder.instances == []
    assert [r.getMessage() for r in caplog.records if "lock" in r.getMessage()] == [
        f"another archive-record run holds the lock on {tmp_path / 'c'}; exiting"
    ]


def test_the_lock_is_released_when_a_cycle_ends(tmp_path: Path) -> None:
    arguments = ["--store-root", str(tmp_path / "s"), "--cache-dir", str(tmp_path / "c")]
    cli.main(arguments)
    cli.main(arguments)
    assert len(_StubRecorder.instances) == 2
