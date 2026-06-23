from datetime import datetime, timezone

import trackllm_website.config as _config_module
from trackllm_website.spend import cumulative_by_kind

# Reference module-level config for monkeypatch
config = _config_module.config


def test_write_lt_spend(monkeypatch, tmp_path):
    monkeypatch.setattr(type(config), "spend_dir", property(lambda self: tmp_path))
    from trackllm_website.main import write_lt_spend

    summary = {"m/a#p": {"success": 9, "error": 1, "total_cost": 0.12}}
    write_lt_spend(summary, datetime(2026, 6, 22, tzinfo=timezone.utc))
    assert abs(cumulative_by_kind(tmp_path)["lt"] - 0.12) < 1e-9
