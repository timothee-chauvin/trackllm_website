import pytest
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


def test_config_loads_without_api_key(monkeypatch):
    """The Pages deploy job has no secrets; the site generator instantiates
    Config at import time and must not require OPENROUTER_API_KEY
    (regression: post-#18 deploy failure, run 28676443013)."""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    cfg = Config(_env_file=None)
    assert cfg.openrouter_api_key is None


def test_require_openrouter_api_key_fails_loud(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    cfg = Config(_env_file=None)
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        cfg.require_openrouter_api_key()
    monkeypatch.setenv("OPENROUTER_API_KEY", "dummy")
    assert Config(_env_file=None).require_openrouter_api_key() == "dummy"
