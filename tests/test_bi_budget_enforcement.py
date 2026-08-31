"""Tests: budget killer enforcement in the lifecycle, monitor, and digest."""

import asyncio
from datetime import datetime, timedelta, timezone


from trackllm_website.bi.digest import MonitorReport, MonitorRow, OnboardingReport
from trackllm_website.bi.reinit import ReinitResult
from trackllm_website.bi.state import EndpointBIState, Epoch
from trackllm_website.config import Endpoint, config
from trackllm_website.spend import Spend, append_entry


def ep(model, cost_per_request):
    return Endpoint(
        api="openrouter",
        model=model,
        provider="p",
        cost=(1, 1),
        cost_per_request=cost_per_request,
    )


def test_config_has_budget_section():
    assert config.budget.hard_cap_per_month > 0
    assert config.budget.target_per_month < config.budget.hard_cap_per_month
    assert config.budget.projection_window_days >= 1


class _FakeClient:
    def __init__(self, *args, **kwargs):
        pass

    async def close(self):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()


def _patch_lifecycle(monkeypatch, tmp_path, *, candidates, projected):
    from trackllm_website import update_endpoints as ue

    monkeypatch.setattr(config.bi, "data_dir", tmp_path)
    monkeypatch.setattr(
        type(config), "spend_dir", property(lambda self: tmp_path / "spend")
    )
    monkeypatch.setattr(ue, "OpenRouterClient", _FakeClient)

    async def fake_resolve_strategies(client, endpoints, policy=None, probe_spend=None):
        return {str(e): None for e in endpoints}, {}

    monkeypatch.setattr(ue, "resolve_strategies", fake_resolve_strategies)
    monkeypatch.setattr(
        ue, "select_monitoring_targets", lambda c, p, m: (c, {e: "test" for e in c}, [])
    )
    monkeypatch.setattr(ue, "fetch_popular_models_safe", lambda top_n: [])
    monkeypatch.setattr(
        ue, "ENDPOINTS_CACHE_BI_PATH", tmp_path / "endpoints_cache_bi.yaml"
    )
    monkeypatch.setattr(ue, "projected_month_end", lambda sd, now, w: projected)
    # The lifecycle uses the real clock; pin the horizon so these tests don't
    # change behavior on the last days of a month (retirement gain ~ days left).
    monkeypatch.setattr(ue, "remaining_days_in_month", lambda now: 10)

    onboarded = []

    async def fake_reinit(client, strategy, endpoint, old_bis, now):
        onboarded.append(endpoint)
        return ReinitResult(
            epoch=Epoch(
                start=now,
                border_inputs=["x"],
                reference={"x": [(now.isoformat(), "t")]},
            ),
            reason="ok",
        )

    monkeypatch.setattr(ue, "reinit", fake_reinit)
    return onboarded


def test_lifecycle_drops_most_expensive_onboards_first(monkeypatch, tmp_path):
    from trackllm_website.update_endpoints import update_endpoints_bi_lifecycle

    dear = ep("m/dear", 1e-4)  # expected onboard ~$2.02
    cheap = ep("m/cheap", 5e-5)  # ~$1.01
    monkeypatch.setattr(config.budget, "hard_cap_per_month", 100.0)
    # projected 98.5 + pending (2.02 + 1.01) = 101.53: dropping "dear" (the
    # most expensive) alone brings it back under the 100 cap.
    onboarded = _patch_lifecycle(
        monkeypatch, tmp_path, candidates=[dear, cheap], projected=98.5
    )

    report = asyncio.run(update_endpoints_bi_lifecycle([dear, cheap]))

    assert [e.model for e in onboarded] == ["m/cheap"]
    skipped = [r for r in report.rows if r.outcome == "skipped_budget"]
    assert [r.model for r in skipped] == ["m/dear"]


def test_lifecycle_under_cap_onboards_everything(monkeypatch, tmp_path):
    from trackllm_website.update_endpoints import update_endpoints_bi_lifecycle

    a, b = ep("m/a", 1e-5), ep("m/b", 1e-5)
    monkeypatch.setattr(config.budget, "hard_cap_per_month", 100.0)
    onboarded = _patch_lifecycle(
        monkeypatch, tmp_path, candidates=[a, b], projected=50.0
    )

    report = asyncio.run(update_endpoints_bi_lifecycle([a, b]))

    assert {e.model for e in onboarded} == {"m/a", "m/b"}
    assert not [r for r in report.rows if r.outcome == "skipped_budget"]


def test_lifecycle_retires_monitored_when_pending_not_enough(monkeypatch, tmp_path):
    from trackllm_website.update_endpoints import update_endpoints_bi_lifecycle

    now = datetime.now(tz=timezone.utc)
    hot = ep("m/hot", 1e-5)
    state = EndpointBIState(
        endpoint=hot,
        status="monitoring",
        epochs=[
            Epoch(start=now - timedelta(days=30), border_inputs=["x"], reference={})
        ],
    )
    monkeypatch.setattr(config.budget, "hard_cap_per_month", 100.0)
    _patch_lifecycle(monkeypatch, tmp_path, candidates=[hot], projected=1000.0)
    state.save(config.bi.state_dir)
    append_entry(
        tmp_path / "spend", state.slug, "monitor", Spend(cost=7.0, n_queries=1), now
    )

    report = asyncio.run(update_endpoints_bi_lifecycle([hot]))

    saved = EndpointBIState.load(config.bi.state_dir / f"{state.slug}.json")
    assert saved.status == "retired"
    assert saved.retired.reason == "budget"
    assert saved.epochs[-1].end_reason == "budget"
    assert [r.outcome for r in report.rows if r.model == "m/hot"] == ["retired_budget"]


def test_monitor_skips_reinit_when_over_budget(monkeypatch, tmp_path):
    from trackllm_website.bi import monitor as monitor_mod
    from trackllm_website.bi.monitor import Decision, run_endpoint

    now = datetime(2026, 8, 20, tzinfo=timezone.utc)
    state = EndpointBIState(
        endpoint=ep("m/x", 1e-5),
        status="monitoring",
        epochs=[
            Epoch(start=now - timedelta(days=5), border_inputs=["x"], reference={})
        ],
    )
    monkeypatch.setattr(config.bi, "data_dir", tmp_path)
    monkeypatch.setattr(
        type(config), "spend_dir", property(lambda self: tmp_path / "spend")
    )

    async def fake_sample_prompts(*args, **kwargs):
        return {"x": [(now.isoformat(), "t")]}, 0

    monkeypatch.setattr(monitor_mod, "sample_prompts", fake_sample_prompts)
    monkeypatch.setattr(monitor_mod, "load_phase2_results", lambda d: {})
    monkeypatch.setattr(
        monitor_mod,
        "decide",
        lambda state, results, now: Decision(
            action="reinit", change_date=now, detector="adaptive"
        ),
    )

    async def forbidden_reinit(*args, **kwargs):
        raise AssertionError("reinit must not run when over budget")

    monkeypatch.setattr(monitor_mod, "reinit", forbidden_reinit)

    rows: list[MonitorRow] = []

    async def go():
        from trackllm_website.api import OpenRouterClient
        from trackllm_website.bi.common import PlainStrategy

        async with OpenRouterClient() as client:
            await run_endpoint(
                client, PlainStrategy(), state, now, event_rows=rows, skip_reinit=True
            )

    asyncio.run(go())

    assert [r.event for r in rows] == ["reinit_skipped_budget"]
    saved = EndpointBIState.load(config.bi.state_dir / f"{state.slug}.json")
    assert saved.current_epoch is not None  # epoch left open, re-detects later


def _seed_ledger(tmp_path, day_cost):
    now = datetime.now(tz=timezone.utc)
    append_entry(tmp_path, "e", "lt", Spend(cost=day_cost, n_queries=1), now)
    return now


def _budget_span(html):
    """The budget header's own span, so row badges can't shadow the assertion."""
    return html.split("projected")[1][:160]


def test_digest_header_red_when_projected_over_cap(monkeypatch, tmp_path):
    from trackllm_website.bi.digest import build_monitoring_email

    now = _seed_ledger(tmp_path, 50.0)
    monkeypatch.setattr(config.budget, "target_per_month", 5.0)
    monkeypatch.setattr(config.budget, "hard_cap_per_month", 10.0)
    report = MonitorReport(date=now.date().isoformat(), rows=[], n_endpoints=1)
    _, plain, html = build_monitoring_email(report, tmp_path)
    assert "projected" in plain
    assert f"cap ${config.budget.hard_cap_per_month:,.2f}" in plain
    assert "#cf222e" in _budget_span(html)


def test_digest_header_amber_when_over_target_under_cap(monkeypatch, tmp_path):
    from trackllm_website.bi.digest import build_monitoring_email

    now = _seed_ledger(tmp_path, 6.0)
    monkeypatch.setattr(config.budget, "target_per_month", 5.0)
    monkeypatch.setattr(config.budget, "hard_cap_per_month", 1000.0)
    report = MonitorReport(date=now.date().isoformat(), rows=[], n_endpoints=1)
    _, plain, html = build_monitoring_email(report, tmp_path)
    assert "over target" in plain
    assert "#bf8700" in _budget_span(html)
    assert "#cf222e" not in _budget_span(html)


def test_digest_header_plain_when_under_target(monkeypatch, tmp_path):
    from trackllm_website.bi.digest import build_onboarding_email

    now = _seed_ledger(tmp_path, 0.01)
    monkeypatch.setattr(config.budget, "target_per_month", 500.0)
    monkeypatch.setattr(config.budget, "hard_cap_per_month", 1000.0)
    report = OnboardingReport(date=now.date().isoformat(), rows=[])
    report.rows = []  # header must render even on an empty report
    _, plain, html = build_onboarding_email(report, tmp_path)
    assert "projected" in plain
    assert "#cf222e" not in _budget_span(html)
    assert "#bf8700" not in _budget_span(html)


def test_onboarding_headline_counts_budget_killer_rows(monkeypatch, tmp_path):
    from trackllm_website.bi.digest import OnboardRow, build_onboarding_email

    now = _seed_ledger(tmp_path, 0.01)
    monkeypatch.setattr(config.budget, "target_per_month", 500.0)
    monkeypatch.setattr(config.budget, "hard_cap_per_month", 1000.0)
    report = OnboardingReport(
        date=now.date().isoformat(),
        rows=[
            OnboardRow("m/a", "p", "skipped_budget", None, 0.0),
            OnboardRow("m/b", "p", "retired_budget", None, 0.0),
            OnboardRow("m/c", "p", "not_selected_budget", None, 0.0),
        ],
    )
    subject, plain, _ = build_onboarding_email(report, tmp_path)
    for text in (subject, plain.split("\n")[1]):
        assert "1 over budget" in text  # selection skip (#92)
        assert "2 budget-killed" in text  # projection killer
