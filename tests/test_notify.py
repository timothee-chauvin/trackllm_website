import pytest

from trackllm_website.notify import build_message, load_creds_from_env

CTX = {
    "workflow": "Run Main Script",
    "run_number": "42",
    "event": "schedule",
    "head_sha": "abc1234def",
    "run_url": "https://github.com/x/y/actions/runs/123",
}


def test_build_message_includes_run_details():
    subject, body = build_message(CTX)
    assert "Run Main Script" in subject
    assert "fail" in subject.lower()
    assert "https://github.com/x/y/actions/runs/123" in body
    assert "42" in body
    assert "abc1234def" in body


def test_load_creds_fails_loud_on_missing(monkeypatch):
    for k in ("GMAIL_USER", "GMAIL_APP_PASSWORD", "NOTIFY_EMAIL"):
        monkeypatch.delenv(k, raising=False)
    with pytest.raises(SystemExit):
        load_creds_from_env()


def test_load_creds_returns_all_when_present(monkeypatch):
    monkeypatch.setenv("GMAIL_USER", "a@b.c")
    monkeypatch.setenv("GMAIL_APP_PASSWORD", "secret")
    monkeypatch.setenv("NOTIFY_EMAIL", "me@b.c")
    creds = load_creds_from_env()
    assert creds == {
        "GMAIL_USER": "a@b.c",
        "GMAIL_APP_PASSWORD": "secret",
        "NOTIFY_EMAIL": "me@b.c",
    }
