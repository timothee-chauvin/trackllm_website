from datetime import datetime, timezone

import trackllm_website.update_endpoints as ue
from trackllm_website.config import Endpoint
from trackllm_website.update_endpoints import save_endpoints_bi


def test_save_endpoints_persists_created(tmp_path, monkeypatch):
    monkeypatch.setattr(
        ue.config, "endpoints_yaml_path_bi", tmp_path / "endpoints_bi.yaml"
    )
    e = Endpoint(
        api="openrouter",
        model="m/x",
        provider="p",
        cost=(1, 1),
        cost_per_request=0.00001,
        created=datetime(2026, 6, 16, tzinfo=timezone.utc),
    )
    save_endpoints_bi([e])
    text = (tmp_path / "endpoints_bi.yaml").read_text()
    assert "created" in text and "2026-06-16" in text
