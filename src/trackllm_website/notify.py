"""Email a notification when a GitHub Actions workflow run fails.

Pure standard library on purpose: this runs from a `workflow_run` watcher on the
system Python, so it must work even when the failing job never got as far as
`uv sync`. Run details arrive via WF_* env vars; credentials via three secrets
(GMAIL_USER, GMAIL_APP_PASSWORD, NOTIFY_EMAIL).
"""

import os
import smtplib
import sys
from email.message import EmailMessage

SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = 465
REQUIRED_KEYS = ("GMAIL_USER", "GMAIL_APP_PASSWORD", "NOTIFY_EMAIL")


def load_creds_from_env() -> dict[str, str]:
    missing = [k for k in REQUIRED_KEYS if not os.environ.get(k)]
    if missing:
        sys.exit(f"notify: missing required secrets: {missing}")
    return {k: os.environ[k] for k in REQUIRED_KEYS}


def build_message(ctx: dict[str, str]) -> tuple[str, str]:
    subject = f"[trackllm] workflow failed: {ctx['workflow']}"
    body = "\n".join(
        [
            f"Workflow: {ctx['workflow']}",
            f"Run:      #{ctx['run_number']}",
            f"Event:    {ctx['event']}",
            f"Commit:   {ctx['head_sha']}",
            f"Details:  {ctx['run_url']}",
        ]
    )
    return subject, body


def send_email(
    creds: dict[str, str], subject: str, plain: str, html: str | None = None
) -> None:
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = creds["GMAIL_USER"]
    msg["To"] = creds["NOTIFY_EMAIL"]
    msg.set_content(plain)
    if html is not None:
        msg.add_alternative(html, subtype="html")
    with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT) as s:
        s.login(creds["GMAIL_USER"], creds["GMAIL_APP_PASSWORD"])
        s.send_message(msg)


def main() -> None:
    ctx = {
        "workflow": os.environ.get("WF_NAME", "(unknown)"),
        "run_number": os.environ.get("WF_RUN_NUMBER", "?"),
        "event": os.environ.get("WF_EVENT", "?"),
        "head_sha": os.environ.get("WF_HEAD_SHA", "?"),
        "run_url": os.environ.get("WF_RUN_URL", "?"),
    }
    creds = load_creds_from_env()
    subject, body = build_message(ctx)
    send_email(creds, subject, body)
    print(f"notify: sent failure email for {ctx['workflow']} run #{ctx['run_number']}")


if __name__ == "__main__":
    main()
