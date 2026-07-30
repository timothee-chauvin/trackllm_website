import json
from pathlib import Path

from conftest import write_month_dir
from trackllm_website.generate_site.lt import discover_lt_endpoints


def _make_lt_endpoint(root: Path, slug: str, model: str, provider: str):
    d = root / slug / "default"
    d.mkdir(parents=True)
    (d / "info.json").write_text(
        json.dumps({"prompt": "hi", "endpoint": {"model": model, "provider": provider}})
    )
    write_month_dir(d, "2026-06", [["24 10:00:00", 0]])


def test_discover_lt_endpoints(tmp_path):
    _make_lt_endpoint(tmp_path, "m2fa23p", "m/a", "p")
    eps = discover_lt_endpoints(tmp_path)
    assert len(eps) == 1
    assert eps[0].model == "m/a"
    assert eps[0].provider == "p"
    assert eps[0].prompts[0].months == ["2026-06"]
