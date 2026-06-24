from datetime import datetime, timezone

from trackllm_website.bi.state import EndpointBIState, Epoch, RetiredInfo
from trackllm_website.config import Endpoint
from trackllm_website.generate_site.b3it import derive_b3it


def _ep():
    return Endpoint(api="openrouter", model="m/a", provider="p", cost=[0.1, 0.2], max_logprobs=None)


def test_retired_no_reference_yields_empty_tv_but_full_timeline():
    state = EndpointBIState(
        endpoint=_ep(),
        status="retired",
        retired=RetiredInfo(
            reason="no_bis",
            since=datetime(2026, 2, 5, tzinfo=timezone.utc),
            last_recheck=datetime(2026, 2, 5, tzinfo=timezone.utc),
        ),
        epochs=[
            Epoch(
                start=datetime(2026, 1, 14, tzinfo=timezone.utc),
                border_inputs=[],
                reference={},
                end=datetime(2026, 2, 5, tzinfo=timezone.utc),
                end_reason="gap",
            )
        ],
    )
    view = derive_b3it(state, {})
    assert view.status == "retired"
    assert view.retired_reason == "no_bis"
    assert view.tv_series == {"dates": [], "values": []}
    assert len(view.epochs) == 1
    assert view.epochs[0]["end_reason"] == "gap"
    assert view.n_bis == 0


def test_monitoring_with_reference_yields_tv_series():
    ref = {"p1": [("2026-06-01T00:00:00Z", "A")] * 10}
    results = {
        "p1": {
            "2026-06-01T00:00:00+00:00": [("2026-06-01T00:00:00Z", "A")] * 10,
            "2026-06-02T00:00:00+00:00": [("2026-06-02T00:00:00Z", "B")] * 10,
        }
    }
    state = EndpointBIState(
        endpoint=_ep(),
        status="monitoring",
        retired=None,
        epochs=[
            Epoch(
                start=datetime(2026, 6, 1, tzinfo=timezone.utc),
                border_inputs=["p1"],
                reference=ref,
            )
        ],
    )
    view = derive_b3it(state, results)
    assert view.status == "monitoring"
    assert view.n_bis == 1
    assert view.tv_series["values"]  # non-empty
    assert view.tv_series["values"][0] > 0  # B vs A => TV 1.0
