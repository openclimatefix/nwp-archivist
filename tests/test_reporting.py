import logging
from collections.abc import Iterator

import pytest
import sentry_sdk
from sentry_sdk.envelope import Envelope
from sentry_sdk.transport import Transport
from support import INIT

from nwp_archivist.reporting import LogReporter, SentryReporter, make_reporter


class _CapturingTransport(Transport):
    """Keeps every envelope the SDK would send."""

    def __init__(self, envelopes: list[Envelope]) -> None:
        super().__init__()
        self.envelopes = envelopes

    def capture_envelope(self, envelope: Envelope) -> None:
        self.envelopes.append(envelope)


def _items(envelopes: list[Envelope], type_name: str) -> list[dict]:
    return [
        item.payload.json
        for envelope in envelopes
        for item in envelope.items
        if item.type == type_name and item.payload.json is not None
    ]


@pytest.fixture
def envelopes() -> Iterator[list[Envelope]]:
    sent: list[Envelope] = []
    sentry_sdk.init(dsn="http://key@localhost/1", transport=_CapturingTransport(sent))
    yield sent
    sentry_sdk.get_client().close()
    sentry_sdk.init()


def test_without_a_dsn_the_reporter_logs_the_fault_at_warning(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.delenv("SENTRY_DSN", raising=False)
    reporter = make_reporter(monitor_slug="m")
    assert isinstance(reporter, LogReporter)
    with caplog.at_level(logging.WARNING):
        reporter.fault(product="icon-d2", init_time=INIT, fault="partial", detail="99 of 100")
    assert caplog.records[0].levelno == logging.WARNING
    message = caplog.records[0].getMessage()
    assert "product=icon-d2" in message
    assert f"init_time={INIT.isoformat()}" in message
    assert "fault=partial" in message
    assert "99 of 100" in message


def test_the_log_reporter_check_in_does_nothing() -> None:
    LogReporter().check_in(ok=True)


def test_an_empty_dsn_counts_as_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SENTRY_DSN", "")
    assert isinstance(make_reporter(monitor_slug="m"), LogReporter)


def test_the_sentry_reporter_tags_the_product_run_and_fault(envelopes: list[Envelope]) -> None:
    SentryReporter(monitor_slug="m").fault(
        product="icon-d2", init_time=INIT, fault="missing", detail="none"
    )
    sentry_sdk.get_client().flush()
    (event,) = _items(envelopes, "event")
    assert event["tags"] == {
        "product": "icon-d2",
        "fault": "missing",
        "init_time": INIT.isoformat(),
    }
    assert event["fingerprint"] == ["missing", "icon-d2", INIT.isoformat()]
    assert event["level"] == "warning"
    assert "icon-d2" in event["message"]


def test_the_sentry_reporter_sends_a_cron_check_in(envelopes: list[Envelope]) -> None:
    SentryReporter(monitor_slug="nwp-archive-mogreps").check_in(ok=True)
    SentryReporter(monitor_slug="nwp-archive-dwd").check_in(ok=False)
    sentry_sdk.get_client().flush()
    statuses = [item["status"] for item in _items(envelopes, "check_in")]
    assert statuses == ["ok", "error"]
    slugs = [item["monitor_slug"] for item in _items(envelopes, "check_in")]
    assert slugs == ["nwp-archive-mogreps", "nwp-archive-dwd"]


def test_a_logged_error_does_not_become_an_untagged_sentry_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sent: list[Envelope] = []
    monkeypatch.setenv("SENTRY_DSN", "http://key@localhost/1")
    reporter = make_reporter(monitor_slug="m", transport=_CapturingTransport(sent))
    assert isinstance(reporter, SentryReporter)
    try:
        logging.getLogger("nwp_archivist.recorder").error("commit failed")
        reporter.fault(product="icon-d2", init_time=INIT, fault="partial", detail="99 of 100")
        sentry_sdk.get_client().flush()
        events = _items(sent, "event")
    finally:
        sentry_sdk.get_client().close()
        sentry_sdk.init()
    assert [event["tags"]["fault"] for event in events] == ["partial"]
