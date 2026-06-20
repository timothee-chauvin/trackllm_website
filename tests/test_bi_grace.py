from datetime import datetime, timedelta, timezone

from trackllm_website.bi.state import EndpointBIState
from trackllm_website.config import Endpoint
from trackllm_website.update_endpoints import should_delist

NOW = datetime(2026, 6, 17, tzinfo=timezone.utc)


def state(model, deselected_since):
    return EndpointBIState(
        endpoint=Endpoint(api="openrouter", model=model, provider="p", cost=(1, 1)),
        status="monitoring",
        epochs=[],
        deselected_since=deselected_since,
    )


def test_grace_logic():
    grace = 30
    assert (
        should_delist(state("a", None), NOW, grace) is False
    )  # just fell out this run
    assert (
        should_delist(state("b", NOW - timedelta(days=10)), NOW, grace) is False
    )  # within grace
    assert (
        should_delist(state("c", NOW - timedelta(days=31)), NOW, grace) is True
    )  # past grace
