"""Fetching and decoding single DWD GRIB2 files.

A file that is absent, truncated, undecodable or not the file its address promised is "not yet":
the caller leaves it unrecorded and asks again on the next cycle. Nothing here raises for the
provider misbehaving.
"""

import logging
import random
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Final

import eccodes
import httpx
import numpy as np

from nwp_archivist.cache import RunGrid
from nwp_archivist.products import ExpectedFile, Product, expected_files, static_urls
from nwp_archivist.source import Cropped, NotYet
from nwp_archivist.store import box_cell_index

logger = logging.getLogger(__name__)

# ecCodes documents its handles as safe to use from separate threads only in some builds, so
# decoding is serialised. A decode takes about 15 ms, whereas a download takes far longer, so
# downloads still run in parallel.
_ECCODES_LOCK: Final[threading.Lock] = threading.Lock()

RETRY_STATUS_CODES: Final[frozenset[int]] = frozenset({408, 425, 429, 500, 502, 503, 504})


@dataclass(frozen=True)
class Expectation:
    """What a decoded message must say, so that a mislabelled file is never archived.

    Attributes:
        short_name: The `shortName` the message must carry, or `None` to skip the check.
        step_minutes: The lead time in minutes the message must carry, or `None` to skip.
        member: The ensemble member the message must carry, or `None` for a deterministic product
            (or a static field, which is not checked).
        level: The model level the message must carry, or `None` to skip the check.
    """

    short_name: str | None = None
    step_minutes: int | None = None
    member: int | None = None
    level: int | None = None


@dataclass(frozen=True)
class Decoded:
    """One decoded message on the provider's native grid.

    Attributes:
        values: The field as `float32`, with cells masked by a GRIB bitmap set to NaN.
        generating_process: The GRIB `generatingProcessIdentifier`, which changes when the weather
            model configuration changes.
    """

    values: np.ndarray
    generating_process: int


class MismatchedFileError(ValueError):
    """The decoded message does not say what the file's address promised."""


def _step_minutes(handle: int) -> int:
    """Read the end of the message's step range in minutes, whatever unit the file uses."""
    eccodes.codes_set(handle, "stepUnits", "m")
    return int(str(eccodes.codes_get(handle, "endStep")).removesuffix("m"))


def decode_grib(body: bytes, *, expect: Expectation) -> Decoded:
    """Decode the first GRIB2 message in `body` and check it against what its address promised.

    Cells masked by a GRIB bitmap, which ecCodes would otherwise return as 9999.0, become NaN.

    Args:
        body: The bytes of one GRIB2 file.
        expect: The header values the message must carry.

    Returns:
        The values as `float32`, and the generating process identifier.

    Raises:
        MismatchedFileError: If the message header disagrees with `expect`.
    """
    with _ECCODES_LOCK:
        handle = eccodes.codes_new_from_message(body)
        try:
            found = {
                "short_name": str(eccodes.codes_get(handle, "shortName")),
                "step_minutes": _step_minutes(handle),
                "member": (
                    int(eccodes.codes_get(handle, "perturbationNumber"))
                    if eccodes.codes_is_defined(handle, "perturbationNumber")
                    else None
                ),
                "level": int(eccodes.codes_get(handle, "level")),
            }
            for key, wanted in (
                ("short_name", expect.short_name),
                ("step_minutes", expect.step_minutes),
                ("member", expect.member),
                ("level", expect.level),
            ):
                if wanted is not None and found[key] != wanted:
                    message = f"{key} is {found[key]!r}, expected {wanted!r}"
                    raise MismatchedFileError(message)
            eccodes.codes_set(handle, "missingValue", float("nan"))
            values = np.asarray(eccodes.codes_get_values(handle), dtype=np.float32)
            generating_process = int(eccodes.codes_get(handle, "generatingProcessIdentifier"))
        finally:
            eccodes.codes_release(handle)
    return Decoded(values=values, generating_process=generating_process)


@dataclass
class Fetcher:
    """Downloads files with keep-alive, retries with jittered backoff, and an optional rate limit.

    Attributes:
        client: The HTTP client, reused for every request so connections stay open.
        max_attempts: How many times to try a file that fails with a retryable error.
        backoff_seconds: The delay before the first retry, doubled for each further retry.
        max_requests_per_second: A cap on the request rate across all threads, or `None`.
        sleep: The function that waits, replaceable so that tests do not wait.
        jitter: A function returning a random number in `[0, 1)`, replaceable for tests.
    """

    client: httpx.Client
    max_attempts: int = 3
    backoff_seconds: float = 1.0
    max_requests_per_second: float | None = None
    sleep: Callable[[float], None] = time.sleep
    jitter: Callable[[], float] = random.random

    def __post_init__(self) -> None:
        """Set up the shared rate limiter state."""
        self._rate_lock = threading.Lock()
        self._next_request_time = 0.0

    def _wait_for_rate_limit(self) -> None:
        """Block until this thread may send a request under `max_requests_per_second`."""
        if self.max_requests_per_second is None:
            return
        with self._rate_lock:
            now = time.monotonic()
            start = max(now, self._next_request_time)
            self._next_request_time = start + 1.0 / self.max_requests_per_second
        if start > now:
            self.sleep(start - now)

    def _request(self, url: str, headers: dict[str, str] | None = None) -> httpx.Response | NotYet:
        """Send a GET request, retrying transient failures.

        Args:
            url: The address of the file.
            headers: Extra request headers, such as `Range`, or `None`.

        Returns:
            The response (status 200 or 206), or `NotYet` when the file is absent (404), a
            retryable error persists, or the body is shorter than its `Content-Length` header says.
        """
        reason = "no attempt made"
        for attempt in range(self.max_attempts):
            if attempt > 0:
                self.sleep(self.backoff_seconds * 2 ** (attempt - 1) * (1 + self.jitter()))
            self._wait_for_rate_limit()
            try:
                response = self.client.get(url, headers=headers)
            except httpx.TransportError as error:
                reason = f"transport error: {type(error).__name__}"
                continue
            if response.status_code == httpx.codes.NOT_FOUND:
                return NotYet("404")
            if response.status_code in RETRY_STATUS_CODES:
                reason = f"http {response.status_code}"
                continue
            if response.status_code not in {httpx.codes.OK, httpx.codes.PARTIAL_CONTENT}:
                return NotYet(f"http {response.status_code}")
            declared = response.headers.get("content-length")
            if declared is not None and int(declared) != len(response.content):
                reason = f"short body: {len(response.content)} of {declared} bytes"
                continue
            return response
        return NotYet(reason)

    def get(self, url: str) -> bytes | NotYet:
        """Download one file, retrying transient failures.

        Args:
            url: The address of the file.

        Returns:
            The body, or `NotYet` when the file is absent (404), a retryable error persists, or
            the body is shorter than its `Content-Length` header says.
        """
        response = self._request(url)
        return response if isinstance(response, NotYet) else response.content

    def get_range(self, url: str, *, start: int, stop: int) -> tuple[bytes, int] | NotYet:
        """Download the bytes `[start, stop)` of one file, retrying transient failures.

        Args:
            url: The address of the file.
            start: The first byte wanted.
            stop: One past the last byte wanted.

        Returns:
            The bytes (fewer than requested if the file ends first) and the size of the whole
            file, or `NotYet` when the file is absent, unreadable, or the response does not say
            how large the file is.
        """
        response = self._request(url, {"Range": f"bytes={start}-{stop - 1}"})
        if isinstance(response, NotYet):
            return response
        total = response.headers.get("content-range", "").rpartition("/")[2]
        if not total.isdigit():
            return NotYet("the response has no Content-Range")
        return response.content, int(total)

    def fetch(self, url: str, *, expect: Expectation) -> Decoded | NotYet:
        """Download and decode one file.

        Args:
            url: The address of the file.
            expect: The header values the message must carry.

        Returns:
            The decoded field, or `NotYet` if the file is absent, truncated, undecodable or
            mislabelled.
        """
        body = self.get(url)
        if isinstance(body, NotYet):
            return body
        try:
            return decode_grib(body, expect=expect)
        except MismatchedFileError as error:
            return NotYet(f"mismatched file: {error}")
        except eccodes.CodesInternalError as error:
            return NotYet(f"undecodable: {error}")


@dataclass(frozen=True)
class DwdSource:
    """The DWD open-data server: one GRIB2 file per variable, member, and lead time.

    Attributes:
        fetcher: Downloads and decodes the files.
        base_url: The root of DWD's directory layout, replaceable so that tests can point elsewhere.
    """

    fetcher: Fetcher
    base_url: str

    def expected_files(self, product: Product, init_time: datetime) -> list[ExpectedFile]:
        """Every file a run should contain."""
        return expected_files(product, init_time, base_url=self.base_url)

    def sequence_key(self, file: ExpectedFile) -> tuple[str, ...]:
        """One variable of one member is a sequence."""
        return (file.field.variable, str(file.member))

    def static_names(self, product: Product) -> set[str]:
        """The grid fields the archive stores once."""
        return {"clat", "clon", "hsurf", "fr_land"} | {f"hhl_L{n}" for n in product.hhl_levels}

    def fetch_grid(
        self,
        *,
        product: Product,
        init_time: datetime,
        only: set[str] | None,
        previous: RunGrid | None,
    ) -> RunGrid | NotYet:
        """Download the static fields and crop them, or return `NotYet` if any is not yet there."""
        urls = static_urls(product, init_time, base_url=self.base_url)
        wanted = {name: url for name, url in urls.items() if only is None or name in only}
        decoded: dict[str, Decoded] = {}
        for name, url in wanted.items():
            level = int(name.removeprefix("hhl_L")) if name.startswith("hhl_L") else None
            outcome = self.fetcher.fetch(url, expect=Expectation(level=level))
            if isinstance(outcome, NotYet):
                logger.info("static %s of %s: %s", name, init_time.isoformat(), outcome.reason)
                return outcome
            decoded[name] = outcome
        clat = decoded["clat"].values
        clon = decoded["clon"].values
        cell_index = box_cell_index(clat=clat, clon=clon)
        statics = {name: values.values[cell_index] for name, values in decoded.items()}
        if previous is not None:
            statics = {**previous.statics, **statics}
        return RunGrid(n_points=len(clat), cell_index=cell_index, statics=statics)

    def fetch(self, file: ExpectedFile, grid: RunGrid) -> Cropped | NotYet:
        """Download, check, and crop one file."""
        expect = Expectation(
            short_name=file.field.short_name,
            step_minutes=file.step_minutes,
            member=file.member,
            level=file.field.level,
        )
        outcome = self.fetcher.fetch(file.url, expect=expect)
        if isinstance(outcome, NotYet):
            return outcome
        # A file of another size is passed on uncropped, so that the recorder sees the mismatch.
        return Cropped(
            values=outcome.values[grid.cell_index]
            if len(outcome.values) == grid.n_points
            else outcome.values,
            n_points=len(outcome.values),
            generating_process=outcome.generating_process,
        )
