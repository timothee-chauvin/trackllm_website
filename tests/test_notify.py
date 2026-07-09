import pytest

from trackllm_website.notify import build_message, load_creds_from_env, send_email

CTX = {
    "workflow": "Run Main Script",
    "run_number": "42",
    "run_attempt": "2",
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
    assert "Attempt:  2" in body


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


def test_send_email_multipart(monkeypatch):
    sent = {}

    class FakeSMTP:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def login(self, *a):
            pass

        def send_message(self, msg):
            sent["msg"] = msg

    monkeypatch.setattr("trackllm_website.notify.smtplib.SMTP_SSL", FakeSMTP)
    send_email(
        {
            "GMAIL_USER": "u@x",
            "GMAIL_APP_PASSWORD": "p",
            "NOTIFY_EMAIL": "to@x",
        },
        "subj",
        "the plain text",
        "<b>hello</b>",
    )
    msg = sent["msg"]
    assert msg["Subject"] == "subj" and msg["To"] == "to@x"
    body = msg.get_body(("html",))
    assert body is not None and "hello" in body.get_content()
