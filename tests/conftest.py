"""Pytest configuration: fixtures shared by the test modules, and the `--run-network` gate.

Tests marked `network` fetch real files from DWD. They are skipped unless `--run-network` is given,
so that a plain `uv run pytest` never touches the provider.
"""

from collections.abc import Iterable, Iterator

import httpx
import pytest
from moto.server import ThreadedMotoServer


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register the `--run-network` opt-in flag."""
    parser.addoption(
        "--run-network",
        action="store_true",
        default=False,
        help="Run tests marked @pytest.mark.network (they hit the real DWD open-data server).",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: Iterable[pytest.Item]) -> None:
    """Skip the network-marked tests unless `--run-network` was given."""
    if config.getoption("--run-network"):
        return
    skip = pytest.mark.skip(reason="needs --run-network")
    for item in items:
        if "network" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def moto_server() -> Iterator[str]:
    """A local S3 server, started once. Icechunk's own S3 client bypasses in-process `mock_aws`."""
    server = ThreadedMotoServer(ip_address="127.0.0.1", port=0, verbose=False)
    server.start()
    host, port = server.get_host_and_port()
    yield f"http://{host}:{port}"
    server.stop()


@pytest.fixture
def s3_endpoint(moto_server: str, monkeypatch: pytest.MonkeyPatch) -> str:
    """The local S3 server, reset before each test so no test sees another's buckets."""
    httpx.post(f"{moto_server}/moto-api/reset").raise_for_status()
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "eu-west-2")
    return moto_server
