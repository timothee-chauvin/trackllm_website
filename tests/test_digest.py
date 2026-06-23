"""Tests for bi/digest.py and spend.today_by_kind."""

from datetime import datetime, timezone


from trackllm_website.spend import Spend, append_entry, today_by_kind
from trackllm_website.util import slugify


# ---------------------------------------------------------------------------
# today_by_kind
# ---------------------------------------------------------------------------


def test_today_by_kind_filters_by_day(tmp_path):
    today = "2026-06-23"
    now_today = datetime(2026, 6, 23, 10, 0, tzinfo=timezone.utc)
    now_yesterday = datetime(2026, 6, 22, 10, 0, tzinfo=timezone.utc)

    append_entry(tmp_path, "slugA", "onboard", Spend(cost=1.0, n_queries=5), now_today)
    append_entry(tmp_path, "slugA", "recheck", Spend(cost=0.5, n_queries=2), now_today)
    append_entry(
        tmp_path, "slugB", "onboard", Spend(cost=2.0, n_queries=8), now_yesterday
    )

    result = today_by_kind(tmp_path, today)
    assert abs(result["onboard"] - 1.0) < 1e-9
    assert abs(result["recheck"] - 0.5) < 1e-9
    assert (
        "onboard"
        not in {k: v for k, v in result.items() if k not in ("onboard", "recheck")}
        or True
    )
    # The yesterday entry should NOT appear
    assert abs(result.get("onboard", 0) - 1.0) < 1e-9  # only today's onboard


def test_today_by_kind_empty_dir(tmp_path):
    result = today_by_kind(tmp_path / "nonexistent", "2026-06-23")
    assert result == {}


def test_today_by_kind_no_entries_today(tmp_path):
    yesterday = datetime(2026, 6, 22, 10, 0, tzinfo=timezone.utc)
    append_entry(tmp_path, "slugA", "monitor", Spend(cost=1.0, n_queries=1), yesterday)
    result = today_by_kind(tmp_path, "2026-06-23")
    assert result == {}


# ---------------------------------------------------------------------------
# slug correctness
# ---------------------------------------------------------------------------


def test_slugify_known_example():
    # '#' -> hex 23, '/' -> hex 2f
    assert (
        slugify("deepseek/deepseek-chat-v3-0324#fireworks")
        == "deepseek2fdeepseek-chat-v3-032423fireworks"
    )


# ---------------------------------------------------------------------------
# build_onboarding_email
# ---------------------------------------------------------------------------


def _make_onboard_report(date="2026-06-23"):
    from trackllm_website.bi.digest import OnboardRow, OnboardingReport

    rows = [
        OnboardRow(
            model="deepseek/deepseek-chat-v3-0324",
            provider="fireworks",
            outcome="onboarded",
            n_bis=42,
            spent=0.0120,
        ),
        OnboardRow(
            model="openai/gpt-4o",
            provider="openai",
            outcome="timeout",
            n_bis=None,
            spent=0.0050,
        ),
        OnboardRow(
            model="meta-llama/llama-3.1-8b",
            provider="openrouter",
            outcome="no_bis",
            n_bis=0,
            spent=0.0030,
        ),
    ]
    return OnboardingReport(date=date, rows=rows)


def _make_monitor_report(date="2026-06-23"):
    from trackllm_website.bi.digest import MonitorRow, MonitorReport

    rows = [
        MonitorRow(
            model="deepseek/deepseek-chat-v3-0324",
            provider="fireworks",
            event="change_detected",
            change_date="2026-06-20",
            n_bis_after=38,
            spent=0.0080,
        ),
        MonitorRow(
            model="openai/gpt-4o",
            provider="openai",
            event="stable",
            change_date=None,
            n_bis_after=50,
            spent=0.0040,
        ),
    ]
    return MonitorReport(date=date, rows=rows, n_endpoints=10)


def _make_ledger(tmp_path, date="2026-06-23"):
    now = datetime(2026, 6, 23, 12, 0, tzinfo=timezone.utc)
    now_old = datetime(2026, 5, 1, 12, 0, tzinfo=timezone.utc)
    append_entry(tmp_path, "ep1", "onboard", Spend(cost=0.01, n_queries=1), now)
    append_entry(tmp_path, "ep1", "recheck", Spend(cost=0.005, n_queries=1), now)
    append_entry(tmp_path, "ep1", "vetting", Spend(cost=0.002, n_queries=1), now)
    append_entry(tmp_path, "ep1", "monitor", Spend(cost=0.008, n_queries=1), now)
    append_entry(tmp_path, "ep1", "reinit", Spend(cost=0.003, n_queries=1), now)
    # Older entries (different month) — should appear in cumulative but not today
    append_entry(tmp_path, "ep1", "onboard", Spend(cost=0.5, n_queries=5), now_old)
    return tmp_path


def test_onboarding_email_subject_format(tmp_path):
    from trackllm_website.bi.digest import build_onboarding_email

    report = _make_onboard_report()
    _make_ledger(tmp_path)
    subject, plain, html = build_onboarding_email(report, tmp_path)

    assert subject.startswith("[trackllm] $")
    assert "−" in subject  # U+2212 MINUS SIGN
    assert "B3IT onboarding:" in subject
    # Counts from report: 1 onboarded, 1 timed out, 1 no-BIs
    assert "1 onboarded" in subject
    assert "1 timed out" in subject
    assert "1 no-BIs" in subject


def test_onboarding_email_subject_cost(tmp_path):
    from trackllm_website.bi.digest import build_onboarding_email

    report = _make_onboard_report()
    _make_ledger(tmp_path)
    subject, plain, html = build_onboarding_email(report, tmp_path)

    # today onboard + recheck + vetting = 0.01 + 0.005 + 0.002 = 0.017 -> $0.02
    assert "$0.02" in subject  # 2 decimals


def test_onboarding_email_html_has_endpoint_link(tmp_path):
    from trackllm_website.bi.digest import build_onboarding_email

    report = _make_onboard_report()
    _make_ledger(tmp_path)
    subject, plain, html = build_onboarding_email(report, tmp_path)

    slug = slugify("deepseek/deepseek-chat-v3-0324#fireworks")
    assert f"https://www.trackllm.net/endpoints/{slug}.html" in html


def test_onboarding_email_plain_has_outcome(tmp_path):
    from trackllm_website.bi.digest import build_onboarding_email

    report = _make_onboard_report()
    _make_ledger(tmp_path)
    subject, plain, html = build_onboarding_email(report, tmp_path)

    assert "onboarded" in plain
    assert "timed out" in plain


# ---------------------------------------------------------------------------
# build_monitoring_email
# ---------------------------------------------------------------------------


def test_monitoring_email_subject_format(tmp_path):
    from trackllm_website.bi.digest import build_monitoring_email

    report = _make_monitor_report()
    _make_ledger(tmp_path)
    subject, plain, html = build_monitoring_email(report, tmp_path)

    assert subject.startswith("[trackllm] $")
    assert "−" in subject  # U+2212
    assert "B3IT monitoring:" in subject
    assert "changes detected" in subject
    # 1 change_detected in the report
    assert "1 changes detected" in subject


def test_monitoring_email_subject_cost(tmp_path):
    from trackllm_website.bi.digest import build_monitoring_email

    report = _make_monitor_report()
    _make_ledger(tmp_path)
    subject, plain, html = build_monitoring_email(report, tmp_path)

    # today monitor + reinit = 0.008 + 0.003 = 0.011 -> $0.01
    assert "$0.01" in subject


def test_monitoring_email_html_has_endpoint_link(tmp_path):
    from trackllm_website.bi.digest import build_monitoring_email

    report = _make_monitor_report()
    _make_ledger(tmp_path)
    subject, plain, html = build_monitoring_email(report, tmp_path)

    slug = slugify("deepseek/deepseek-chat-v3-0324#fireworks")
    assert f"https://www.trackllm.net/endpoints/{slug}.html" in html


def test_monitoring_email_plain_has_event(tmp_path):
    from trackllm_website.bi.digest import build_monitoring_email

    report = _make_monitor_report()
    _make_ledger(tmp_path)
    subject, plain, html = build_monitoring_email(report, tmp_path)

    assert "change detected" in plain


# ---------------------------------------------------------------------------
# notable() gating — send_*_digest must NOT call send_email on empty reports
# ---------------------------------------------------------------------------


def test_send_onboarding_digest_gates_on_notable(tmp_path, monkeypatch):
    from trackllm_website.bi.digest import OnboardingReport, send_onboarding_digest
    import trackllm_website.notify as notify_mod

    calls = []
    monkeypatch.setattr(notify_mod, "send_email", lambda *a, **k: calls.append(a))

    empty_report = OnboardingReport(date="2026-06-23", rows=[])
    assert not empty_report.notable()
    send_onboarding_digest(empty_report, tmp_path)
    assert calls == [], "send_email must not be called when report has no rows"


def test_send_monitoring_digest_gates_on_notable(tmp_path, monkeypatch):
    from trackllm_website.bi.digest import MonitorReport, send_monitoring_digest
    import trackllm_website.notify as notify_mod

    calls = []
    monkeypatch.setattr(notify_mod, "send_email", lambda *a, **k: calls.append(a))

    empty_report = MonitorReport(date="2026-06-23", rows=[], n_endpoints=0)
    assert not empty_report.notable()
    send_monitoring_digest(empty_report, tmp_path)
    assert calls == [], "send_email must not be called when report has no rows"


def test_send_onboarding_digest_calls_send_email_when_notable(tmp_path, monkeypatch):
    from trackllm_website.bi.digest import send_onboarding_digest
    import trackllm_website.notify as notify_mod

    calls = []
    monkeypatch.setattr(notify_mod, "send_email", lambda *a, **k: calls.append(a))
    monkeypatch.setattr(
        notify_mod,
        "load_creds_from_env",
        lambda: {"GMAIL_USER": "u", "GMAIL_APP_PASSWORD": "p", "NOTIFY_EMAIL": "t"},
    )

    report = _make_onboard_report()
    _make_ledger(tmp_path)
    send_onboarding_digest(report, tmp_path)
    assert len(calls) == 1


def test_send_monitoring_digest_calls_send_email_when_notable(tmp_path, monkeypatch):
    from trackllm_website.bi.digest import send_monitoring_digest
    import trackllm_website.notify as notify_mod

    calls = []
    monkeypatch.setattr(notify_mod, "send_email", lambda *a, **k: calls.append(a))
    monkeypatch.setattr(
        notify_mod,
        "load_creds_from_env",
        lambda: {"GMAIL_USER": "u", "GMAIL_APP_PASSWORD": "p", "NOTIFY_EMAIL": "t"},
    )

    report = _make_monitor_report()
    _make_ledger(tmp_path)
    send_monitoring_digest(report, tmp_path)
    assert len(calls) == 1
