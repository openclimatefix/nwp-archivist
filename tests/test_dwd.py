import math
from collections.abc import Callable

import httpx
import numpy as np
from support import make_grib

from nwp_archivist.dwd import Decoded, Expectation, Fetcher, NotYet, decode_grib

URL = "https://provider.test/file.grib2"


def _fetcher(handler: Callable[[httpx.Request], httpx.Response]) -> Fetcher:
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return Fetcher(client=client, sleep=lambda _seconds: None, jitter=lambda: 0.0)


def _values() -> np.ndarray:
    return np.array([280.0, 281.0, 282.0, 283.0], dtype=np.float32)


def test_a_404_is_not_yet() -> None:
    assert _fetcher(lambda _request: httpx.Response(404)).fetch(URL, expect=Expectation()) == (
        NotYet("404")
    )


def test_a_file_that_appears_later_decodes() -> None:
    responses = [httpx.Response(404), httpx.Response(200, content=make_grib(_values()))]
    fetcher = _fetcher(lambda _request: responses.pop(0))
    assert isinstance(fetcher.fetch(URL, expect=Expectation()), NotYet)
    outcome = fetcher.fetch(URL, expect=Expectation())
    assert isinstance(outcome, Decoded)
    np.testing.assert_array_equal(outcome.values, _values())


def test_a_500_is_retried_once_then_succeeds() -> None:
    responses = [httpx.Response(500), httpx.Response(200, content=make_grib(_values()))]
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return responses.pop(0)

    assert isinstance(_fetcher(handler).fetch(URL, expect=Expectation()), Decoded)
    assert len(calls) == 2


def test_a_persistent_500_is_not_yet_after_the_attempts_run_out() -> None:
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(503)

    assert _fetcher(handler).fetch(URL, expect=Expectation()) == NotYet("http 503")
    assert len(calls) == 3


def test_backoff_doubles_between_retries() -> None:
    sleeps: list[float] = []
    client = httpx.Client(transport=httpx.MockTransport(lambda _r: httpx.Response(500)))
    fetcher = Fetcher(client=client, sleep=sleeps.append, jitter=lambda: 0.0, backoff_seconds=2.0)
    fetcher.get(URL)
    assert sleeps == [2.0, 4.0]


def test_a_transport_error_is_retried_then_not_yet() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    assert _fetcher(handler).fetch(URL, expect=Expectation()) == NotYet(
        "transport error: ConnectError"
    )


def test_a_body_shorter_than_content_length_is_not_yet() -> None:
    body = make_grib(_values())
    response = httpx.Response(200, content=body[:-20], headers={"content-length": str(len(body))})
    outcome = _fetcher(lambda _request: response).fetch(URL, expect=Expectation())
    assert isinstance(outcome, NotYet)
    assert "short body" in outcome.reason


def test_a_truncated_message_is_not_yet_even_without_a_length_header() -> None:
    body = make_grib(_values())[:-20]
    outcome = _fetcher(lambda _r: httpx.Response(200, content=body)).fetch(
        URL, expect=Expectation()
    )
    assert isinstance(outcome, NotYet)


def test_garbage_is_not_yet() -> None:
    outcome = _fetcher(lambda _r: httpx.Response(200, content=b"<html>oops</html>")).fetch(
        URL, expect=Expectation()
    )
    assert isinstance(outcome, NotYet)


def test_a_wrong_short_name_is_not_yet() -> None:
    body = make_grib(_values(), short_name="10u")
    outcome = _fetcher(lambda _r: httpx.Response(200, content=body)).fetch(
        URL, expect=Expectation(short_name="2t")
    )
    assert isinstance(outcome, NotYet)
    assert "short_name" in outcome.reason


def test_a_wrong_step_is_not_yet() -> None:
    body = make_grib(_values(), step_minutes=60)
    outcome = _fetcher(lambda _r: httpx.Response(200, content=body)).fetch(
        URL, expect=Expectation(step_minutes=120)
    )
    assert isinstance(outcome, NotYet)
    assert "step_minutes" in outcome.reason


def test_a_wrong_member_is_not_yet() -> None:
    body = make_grib(_values(), member=2)
    outcome = _fetcher(lambda _r: httpx.Response(200, content=body)).fetch(
        URL, expect=Expectation(member=3)
    )
    assert isinstance(outcome, NotYet)
    assert "member" in outcome.reason


def test_a_wrong_level_is_not_yet() -> None:
    body = make_grib(_values(), short_name="u", level=62)
    outcome = _fetcher(lambda _r: httpx.Response(200, content=body)).fetch(
        URL, expect=Expectation(level=63)
    )
    assert isinstance(outcome, NotYet)
    assert "level" in outcome.reason


def test_a_matching_header_passes_every_check() -> None:
    body = make_grib(_values(), short_name="u", level=63, member=4, step_minutes=90)
    decoded = decode_grib(
        body, expect=Expectation(short_name="u", step_minutes=90, member=4, level=63)
    )
    assert decoded.values.dtype == np.float32


def test_a_bitmapped_cell_decodes_to_nan_never_9999() -> None:
    values = _values()
    values[1] = np.nan
    decoded = decode_grib(make_grib(values), expect=Expectation())
    assert math.isnan(decoded.values[1])
    assert not (decoded.values == 9999).any()
    np.testing.assert_array_equal(decoded.values[[0, 2, 3]], values[[0, 2, 3]])


def test_the_rate_limit_spaces_requests() -> None:
    sleeps: list[float] = []
    client = httpx.Client(transport=httpx.MockTransport(lambda _r: httpx.Response(404)))
    fetcher = Fetcher(client=client, sleep=sleeps.append, max_requests_per_second=2.0)
    for _ in range(3):
        fetcher.get(URL)
    assert len(sleeps) == 2
    # The fake sleep returns at once, so each wait is measured from the same start.
    assert 0.4 < sleeps[0] <= 0.5
    assert 0.9 < sleeps[1] <= 1.0
