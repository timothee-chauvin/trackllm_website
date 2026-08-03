import json

from trackllm_website.bi.state import EndpointBIState
from trackllm_website.bi.vetting import EndpointCache, UnprobeableEntry
from trackllm_website.config import Endpoint
from trackllm_website.generate_site import spend as spend_mod
from trackllm_website.generate_site.lt import EndpointInfo
from trackllm_website.generate_site.spend import (
    GROUP_LABEL,
    GROUP_ORDER,
    aggregate_spend,
    build_endpoint_names,
    group_for_kind,
)


def test_group_for_kind():
    assert group_for_kind("reinit") == "onboarding"
    assert group_for_kind("monitor") == "monitoring"
    assert group_for_kind("lt") == "lt"
    assert group_for_kind("zzz") == "other"


def _line(ts, kind, cost):
    return json.dumps(
        {"timestamp": ts, "kind": kind, "cost": cost, "n_queries": 1, "n_errors": 0}
    )


def test_aggregate(tmp_path):
    # s1: onboarding + monitoring (May), plus earlier monitor (Jun 10) and recent monitor (Jun 24)
    d = tmp_path / "s1"
    d.mkdir(parents=True)
    (d / "2026-06.jsonl").write_text(
        _line("2026-06-24T00:00:00Z", "onboard", 0.10)
        + "\n"
        + _line("2026-06-24T00:00:00Z", "monitor", 0.02)
        + "\n"
        + _line("2026-06-10T00:00:00Z", "monitor", 0.01)
        + "\n"
        + _line("2026-05-01T00:00:00Z", "vetting", 0.01)
        + "\n"
    )
    # s2: recent lt (smaller total cost than s1)
    d = tmp_path / "s2"
    d.mkdir(parents=True)
    (d / "2026-06.jsonl").write_text(_line("2026-06-20T00:00:00Z", "lt", 0.03) + "\n")

    out = aggregate_spend(tmp_path, "2026-06-24", {"s1": "org/model @ provider"})

    # Existing assertions (with updated monitoring total: 0.03 instead of 0.02)
    assert round(out["cumulative"]["onboarding"], 2) == 0.10
    assert round(out["cumulative"]["monitoring"], 2) == 0.03
    assert (
        round(out["last_30d"].get("vetting", 0), 2) == 0.0
    )  # May 1 is >30d before Jun 24
    assert out["by_endpoint"][0]["slug"] == "s1"
    assert any(day["date"] == "2026-06-24" for day in out["daily"])

    # New assertions: 30d inclusion (recent lt and monitor within 30 days)
    assert round(out["last_30d"].get("lt", 0), 2) == 0.03
    assert round(out["last_30d"].get("monitoring", 0), 2) == 0.03

    # daily is ascending by date
    dates = [d["date"] for d in out["daily"]]
    assert len(dates) >= 2, f"Expected at least 2 dates, got {len(dates)}"
    assert dates == sorted(dates), f"Dates not sorted: {dates}"

    # by_endpoint is descending by total; s1 (0.16) should precede s2 (0.03)
    totals = [e["total"] for e in out["by_endpoint"]]
    assert len(totals) >= 2, f"Expected at least 2 endpoints, got {len(totals)}"
    assert totals == sorted(totals, reverse=True), f"Totals not descending: {totals}"
    assert out["by_endpoint"][0]["slug"] == "s1"
    assert out["by_endpoint"][1]["slug"] == "s2"

    # A resolvable slug gets its human-readable name; an unresolved one (s2, not
    # in `names`) falls back to the slug itself rather than showing nothing.
    assert out["by_endpoint"][0]["name"] == "org/model @ provider"
    assert out["by_endpoint"][1]["name"] == "s2"


def test_group_order_emitted_and_zero_cost_groups_kept(tmp_path):
    d = tmp_path / "s0"
    d.mkdir(parents=True)
    (d / "2026-06.jsonl").write_text(_line("2026-06-24T00:00:00Z", "lt", 0.0) + "\n")
    out = aggregate_spend(tmp_path, "2026-06-24", {})
    assert out["group_order"] == ["onboarding", "monitoring", "lt", "vetting", "other"]
    # A zero-cost run is data ("billed $0"), distinct from "no data": the key stays.
    assert out["by_endpoint"][0]["groups"]["lt"] == 0.0


def test_group_label_covers_every_group_and_renames_per_spec():
    # The exact rename the spend page asked for (never the raw internal keys).
    assert GROUP_LABEL["onboarding"] == "B3IT (onboarding)"
    assert GROUP_LABEL["monitoring"] == "B3IT (monitoring)"
    assert GROUP_LABEL["lt"] == "LT"
    assert GROUP_LABEL["vetting"] == "Vetting"
    assert set(GROUP_ORDER) <= set(GROUP_LABEL)


def test_build_endpoint_names_resolves_from_discovered_and_bi_state(
    tmp_path, monkeypatch
):
    # Isolate from the real repo's BI catalogs / reject cache, so this only
    # exercises the `discovered` and `bi_states` sources.
    monkeypatch.setattr(spend_mod.config, "endpoints_bi", [])
    monkeypatch.setattr(spend_mod.config, "endpoints_bi_prevalence", [])
    monkeypatch.setattr(
        spend_mod, "ENDPOINTS_CACHE_BI_PATH", tmp_path / "no-such-cache.yaml"
    )

    discovered = [
        EndpointInfo(model="m/a", provider="p", slug="m2fa23p", prompts=[]),
    ]
    bi_states = {
        "m2fb23p": EndpointBIState(
            endpoint=Endpoint(api="openrouter", model="m/b", provider="p", cost=(1, 1)),
            status="monitoring",
            epochs=[],
        )
    }

    names = build_endpoint_names(discovered, bi_states)
    assert names == {"m2fa23p": "m/a @ p", "m2fb23p": "m/b @ p"}
    assert "unknown-slug" not in names


def test_build_endpoint_names_uses_reject_cache_when_never_onboarded(
    tmp_path, monkeypatch
):
    """A liar/too-expensive/unprobeable endpoint was probed but never onboarded,
    so it has no BI state -- the reject cache is its only source of a name."""
    monkeypatch.setattr(spend_mod.config, "endpoints_bi", [])
    monkeypatch.setattr(spend_mod.config, "endpoints_bi_prevalence", [])
    cache_path = tmp_path / "endpoints_cache_bi.yaml"
    monkeypatch.setattr(spend_mod, "ENDPOINTS_CACHE_BI_PATH", cache_path)

    liar = Endpoint(api="openrouter", model="m/liar", provider="p", cost=(1, 1))
    unprobeable = Endpoint(api="openrouter", model="m/flaky", provider="p", cost=(1, 1))
    EndpointCache(
        liars=[liar],
        too_expensive=[],
        bad_temperature=[],
        unprobeable=[
            UnprobeableEntry(endpoint=unprobeable, reason="flaky", detail="boom")
        ],
    ).save(cache_path)

    names = build_endpoint_names([], {})
    assert names["m2fliar23p"] == "m/liar @ p"
    assert names["m2fflaky23p"] == "m/flaky @ p"
