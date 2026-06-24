from trackllm_website.generate_site.spend import aggregate_spend, group_for_kind


def test_group_for_kind():
    assert group_for_kind("reinit") == "onboarding"
    assert group_for_kind("monitor") == "monitoring"
    assert group_for_kind("lt") == "lt"
    assert group_for_kind("zzz") == "other"


def _line(ts, kind, cost):
    import json
    return json.dumps({"timestamp": ts, "kind": kind, "cost": cost, "n_queries": 1, "n_errors": 0})


def test_aggregate(tmp_path):
    d = tmp_path / "s1"; d.mkdir(parents=True)
    (d / "2026-06.jsonl").write_text(
        _line("2026-06-24T00:00:00Z", "onboard", 0.10) + "\n" +
        _line("2026-06-24T00:00:00Z", "monitor", 0.02) + "\n" +
        _line("2026-05-01T00:00:00Z", "vetting", 0.01) + "\n")
    out = aggregate_spend(tmp_path, "2026-06-24")
    assert round(out["cumulative"]["onboarding"], 2) == 0.10
    assert round(out["cumulative"]["monitoring"], 2) == 0.02
    assert round(out["last_30d"].get("vetting", 0), 2) == 0.0  # May 1 is >30d before Jun 24
    assert out["by_endpoint"][0]["slug"] == "s1"
    assert any(day["date"] == "2026-06-24" for day in out["daily"])
