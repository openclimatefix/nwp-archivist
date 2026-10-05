import re
from pathlib import Path
from typing import Final

DEPLOY_DIR: Final[Path] = Path(__file__).parent.parent / "deploy"
SERVICES: Final[tuple[str, ...]] = (
    "nwp-archive.service",
    "nwp-archive-mogreps.service",
    "nwp-archive-mogreps-g.service",
)
MET_OFFICE_SERVICES: Final[tuple[str, ...]] = (
    "nwp-archive-mogreps.service",
    "nwp-archive-mogreps-g.service",
)


def _setting(unit: str, key: str) -> str:
    text = (DEPLOY_DIR / unit).read_text()
    match = re.search(rf"^{key}=(.+)$", text, flags=re.MULTILINE)
    assert match is not None, f"{unit} has no {key}="
    return match.group(1)


def _flag(unit: str, flag: str) -> str | None:
    match = re.search(rf"{flag} (\S+)", _setting(unit, "ExecStart"))
    return match.group(1) if match else None


def test_every_service_has_a_timer() -> None:
    for service in SERVICES:
        assert (DEPLOY_DIR / service.replace(".service", ".timer")).exists()


def test_working_directory_holds_the_executable() -> None:
    for service in SERVICES:
        working_directory = _setting(service, "WorkingDirectory")
        assert _setting(service, "ExecStart").startswith(f"{working_directory}/.venv/bin/")


def test_each_service_has_its_own_cache_directory() -> None:
    cache_dirs = [_flag(service, "--cache-dir") for service in SERVICES]
    assert None not in cache_dirs
    assert len(set(cache_dirs)) == len(SERVICES)


def test_met_office_services_share_one_urgency_file() -> None:
    urgency_files = {_flag(service, "--backfill-urgency-file") for service in MET_OFFICE_SERVICES}
    assert len(urgency_files) == 1
    assert None not in urgency_files


def test_dwd_service_does_not_use_the_urgency_file() -> None:
    assert _flag("nwp-archive.service", "--backfill-urgency-file") is None


def test_met_office_timers_start_the_next_cycle_one_minute_after_the_last() -> None:
    for service in MET_OFFICE_SERVICES:
        timer = (DEPLOY_DIR / service.replace(".service", ".timer")).read_text()
        assert "OnUnitInactiveSec=1min" in timer
