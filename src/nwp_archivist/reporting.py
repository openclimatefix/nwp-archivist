"""Reporting faults and liveness, to Sentry when a DSN is configured and to the log otherwise.

The recorder talks to a `Reporter`, so tests inject a fake and a workstation without a Sentry
account still logs every event.
"""

import logging
import os
from datetime import datetime
from typing import Literal, Protocol

import sentry_sdk
from sentry_sdk.crons import capture_checkin
from sentry_sdk.integrations.logging import LoggingIntegration
from sentry_sdk.transport import Transport

logger = logging.getLogger(__name__)

FaultType = Literal[
    "partial",
    "missing",
    "grid_changed",
    "commit_failed",
    "run_error",
    "disk_low",
    "grid_unchecked",
]


class Reporter(Protocol):
    """Where the recorder sends faults and its once-per-cycle liveness signal."""

    def fault(
        self, *, product: str, init_time: datetime | None, fault: FaultType, detail: str
    ) -> None:
        """Report one fault, naming the product and run at fault."""
        ...

    def check_in(self, *, ok: bool) -> None:
        """Report that a recording cycle finished, and whether it finished cleanly."""
        ...


class LogReporter:
    """Logs each fault at WARNING; the cron check-in does nothing."""

    def fault(
        self, *, product: str, init_time: datetime | None, fault: FaultType, detail: str
    ) -> None:
        """Log the fault with the same fields a Sentry event would carry."""
        logger.warning(
            "fault product=%s init_time=%s fault=%s detail=%s",
            product,
            init_time.isoformat() if init_time else "-",
            fault,
            detail,
        )

    def check_in(self, *, ok: bool) -> None:
        """Do nothing, because no monitor is listening."""


class SentryReporter:
    """Sends each fault to Sentry as a warning tagged with its product, run, and fault type."""

    def __init__(self, *, monitor_slug: str) -> None:
        """Build a reporter whose cron check-ins go to the Sentry monitor `monitor_slug`."""
        self.monitor_slug = monitor_slug

    def fault(
        self, *, product: str, init_time: datetime | None, fault: FaultType, detail: str
    ) -> None:
        """Send one Sentry event, and also log it.

        The fingerprint groups repeats of one fault for one run, so an alert rule fires once.
        """
        logger.warning(
            "fault product=%s init_time=%s fault=%s detail=%s",
            product,
            init_time.isoformat() if init_time else "-",
            fault,
            detail,
        )
        with sentry_sdk.new_scope() as scope:
            scope.set_tag("product", product)
            scope.set_tag("fault", fault)
            if init_time is not None:
                scope.set_tag("init_time", init_time.isoformat())
            init_label = init_time.isoformat() if init_time else "-"
            scope.fingerprint = [fault, product, init_label]
            sentry_sdk.capture_message(
                f"{product} {init_label}: {fault}: {detail}", level="warning"
            )

    def check_in(self, *, ok: bool) -> None:
        """Send a Sentry Crons check-in, so a dead recorder becomes visible."""
        capture_checkin(monitor_slug=self.monitor_slug, status="ok" if ok else "error")


def make_reporter(*, monitor_slug: str, transport: Transport | None = None) -> Reporter:
    """Build the reporter the environment asks for.

    Args:
        monitor_slug: The Sentry cron monitor of this service. Each service has its own, so that a
            live service cannot keep a dead one's monitor green.
        transport: A replacement for Sentry's network transport, for tests.

    Returns:
        A `SentryReporter` (after initialising the SDK) if `SENTRY_DSN` is set and non-empty,
        otherwise a `LogReporter`.
    """
    dsn = os.environ.get("SENTRY_DSN", "")
    if not dsn:
        return LogReporter()
    # Logged errors would become a second event with none of the product, run or fault tags, so
    # only `SentryReporter` sends events.
    sentry_sdk.init(
        dsn=dsn,
        traces_sample_rate=0.0,
        integrations=[LoggingIntegration(event_level=None)],
        transport=transport,
    )
    return SentryReporter(monitor_slug=monitor_slug)
