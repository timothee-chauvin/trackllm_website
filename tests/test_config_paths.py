from pathlib import Path

from trackllm_website.config import Config


def test_lt_dir_is_lt_subdir_of_data_dir(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "dummy")
    cfg = Config()
    assert cfg.data_dir == Path("website/data")
    assert cfg.lt_dir == cfg.data_dir / "lt"
