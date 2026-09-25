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


def test_the_store_root_comes_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ARCHIVE_STORE_ROOT", "s3://bucket/prefix")
    assert cli.main([]) == 0
    assert _StubRecorder.instances[0].config.store.root == "s3://bucket/prefix"


def test_the_flags_reach_the_recorder_and_all_products_run_by_default(tmp_path: Path) -> None:
    code = cli.main(
        ["--store-root", str(tmp_path / "s"), "--cache-dir", str(tmp_path / "c"), "--workers", "3"]
    )
    assert code == 0
    recorder = _StubRecorder.instances[0]
    assert recorder.config.cache_dir == tmp_path / "c"
    assert recorder.config.workers == 3
    assert {p.name for p in recorder.products} == set(PRODUCTS)


def test_the_products_flag_selects_products(tmp_path: Path) -> None:
    cli.main(["--store-root", str(tmp_path), "--products", "icon-d2"])
    assert [p.name for p in _StubRecorder.instances[0].products] == ["icon-d2"]


def test_the_default_cache_directory_is_on_the_data_disk() -> None:
    assert cli.build_parser().parse_args([]).cache_dir == "/mnt/data/nwp-archive-cache"
