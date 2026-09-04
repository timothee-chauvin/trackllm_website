import json

from trackllm_website.generate_site.spend import (
    GROUP_LABEL,
    GROUP_ORDER,
    aggregate_spend,
    group_for_kind,
)


def test_group_for_kind():
    assert group_for_kind("reinit") == "onboarding"
    assert group_for_kind("monitor") == "monitoring"
    assert group_for_kind("lt") == "lt"
    assert group_for_kind("zzz") == "other"


def _line(ts, kind, cost, n_queries=1):
    return json.dumps(
        {
            "timestamp": ts,
            "kind": kind,
            "cost": cost,
            "n_queries": n_queries,
            "n_errors": 0,
        }
    )


def test_aggregate(tmp_path):
    # s1: onboarding + monitoring (Jun 24), an earlier monitor (Jun 10), and an old vetting (May 1)
    d = tmp_path / "s1"
    d.mkdir(parents=True)
    (d / "2026-06.jsonl").write_text(
        _line("2026-06-24T00:00:00Z", "onboard", 0.10, n_queries=200)
        + "\n"
        + _line("2026-06-24T00:00:00Z", "monitor", 0.02)
        + "\n"
        + _line("2026-06-10T00:00:00Z", "monitor", 0.01)
        + "\n"
        + _line("2026-05-01T00:00:00Z", "vetting", 0.01)
        + "\n"
    )
    d = tmp_path / "s2"
    d.mkdir(parents=True)
    (d / "2026-06.jsonl").write_text(_line("2026-06-20T00:00:00Z", "lt", 0.03) + "\n")

    out = aggregate_spend(tmp_path, "2026-06-24")

    assert round(out["cumulative"]["onboarding"], 2) == 0.10
    assert round(out["cumulative"]["monitoring"], 2) == 0.03
    # May 1 is >30d before Jun 24
    assert round(out["last_30d"].get("vetting", 0), 2) == 0.0
    assert round(out["last_30d"].get("lt", 0), 2) == 0.03
    assert round(out["last_30d"].get("monitoring", 0), 2) == 0.03

    s1 = out["by_endpoint"]["s1"]
    assert list(s1["groups"]) == [
        "onboarding",
        "monitoring",
        "vetting",
    ]  # display order
    assert round(s1["total"], 2) == 0.14
    assert round(s1["last_30d"], 2) == 0.13
    assert s1["n_queries"] == 203
    assert s1["since"] == "2026-05-01"
    assert out["by_endpoint"]["s2"]["since"] == "2026-06-20"


def test_group_order_emitted_and_zero_cost_groups_kept(tmp_path):
    d = tmp_path / "s0"
    d.mkdir(parents=True)
    (d / "2026-06.jsonl").write_text(_line("2026-06-24T00:00:00Z", "lt", 0.0) + "\n")
    out = aggregate_spend(tmp_path, "2026-06-24")
    assert out["group_order"] == ["onboarding", "monitoring", "lt", "vetting", "other"]
    # A zero-cost run is data ("billed $0"), distinct from "no data": the key stays.
    assert out["by_endpoint"]["s0"]["groups"]["lt"] == 0.0


def test_group_label_covers_every_group_and_renames_per_spec():
    # Reader-facing labels, never the raw internal keys.
    assert GROUP_LABEL["onboarding"] == "B3IT (onboarding)"
    assert GROUP_LABEL["monitoring"] == "B3IT (monitoring)"
    assert GROUP_LABEL["lt"] == "LT"
    assert GROUP_LABEL["vetting"] == "Vetting"
    assert set(GROUP_ORDER) <= set(GROUP_LABEL)
