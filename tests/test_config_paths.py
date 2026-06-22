from pathlib import Path

from trackllm_website.config import Config


def test_lt_dir_is_lt_subdir_of_data_dir(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "dummy")
    cfg = Config()
    assert cfg.data_dir == Path("website/data")
    assert cfg.lt_dir == cfg.data_dir / "lt"


def test_bi_data_dir_under_website_data(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "dummy")
    cfg = Config()
    assert cfg.bi.data_dir == Path("website/data/b3it")
    assert cfg.bi.state_dir == cfg.bi.data_dir / "state"
    assert cfg.bi.phase_2_dir == cfg.bi.data_dir / "phase_2"
