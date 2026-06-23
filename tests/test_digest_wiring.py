"""Tests for wiring digest senders into the daily entrypoints."""

from unittest.mock import AsyncMock


from trackllm_website.bi.digest import (
    MonitorReport,
    MonitorRow,
    OnboardRow,
    OnboardingReport,
    send_monitoring_digest,
    send_onboarding_digest,
)


# ---------------------------------------------------------------------------
# Gate tests: empty vs non-empty reports
# ---------------------------------------------------------------------------


def test_send_onboarding_digest_gate_empty(tmp_path, monkeypatch):
    import trackllm_website.notify as notify_mod

    calls = []
    monkeypatch.setattr(notify_mod, "send_email", lambda *a, **k: calls.append(a))

    report = OnboardingReport(date="2026-06-23", rows=[])
    send_onboarding_digest(report, tmp_path)
    assert calls == []


def test_send_onboarding_digest_gate_notable(tmp_path, monkeypatch):
    import trackllm_website.notify as notify_mod

    calls = []
    monkeypatch.setattr(notify_mod, "send_email", lambda *a, **k: calls.append(a))
    monkeypatch.setattr(
        notify_mod,
        "load_creds_from_env",
        lambda: {"GMAIL_USER": "u", "GMAIL_APP_PASSWORD": "p", "NOTIFY_EMAIL": "t"},
    )

    report = OnboardingReport(
        date="2026-06-23",
        rows=[OnboardRow("m/x", "prov", "onboarded", 5, 0.01)],
    )
    send_onboarding_digest(report, tmp_path)
    assert len(calls) == 1


def test_send_monitoring_digest_gate_empty(tmp_path, monkeypatch):
    import trackllm_website.notify as notify_mod

    calls = []
    monkeypatch.setattr(notify_mod, "send_email", lambda *a, **k: calls.append(a))

    report = MonitorReport(date="2026-06-23", rows=[], n_endpoints=5)
    send_monitoring_digest(report, tmp_path)
    assert calls == []


def test_send_monitoring_digest_gate_notable(tmp_path, monkeypatch):
    import trackllm_website.notify as notify_mod

    calls = []
    monkeypatch.setattr(notify_mod, "send_email", lambda *a, **k: calls.append(a))
    monkeypatch.setattr(
        notify_mod,
        "load_creds_from_env",
        lambda: {"GMAIL_USER": "u", "GMAIL_APP_PASSWORD": "p", "NOTIFY_EMAIL": "t"},
    )

    report = MonitorReport(
        date="2026-06-23",
        rows=[MonitorRow("m/x", "prov", "reonboarded", "2026-06-20", 10, 0.01)],
        n_endpoints=5,
    )
    send_monitoring_digest(report, tmp_path)
    assert len(calls) == 1


# ---------------------------------------------------------------------------
# monitor.main wires send_monitoring_digest with the report from monitor()
# ---------------------------------------------------------------------------


def test_monitor_main_calls_send_monitoring_digest(tmp_path, monkeypatch):
    """monitor.main() must call send_monitoring_digest with the report returned by monitor()."""
    from trackllm_website.bi import monitor as monitor_mod
    import trackllm_website.bi.digest as digest_mod
    from trackllm_website.config import config

    expected_report = MonitorReport(
        date="2026-06-23",
        rows=[MonitorRow("m/x", "prov", "reonboarded", "2026-06-20", 10, 0.01)],
        n_endpoints=3,
    )

    monkeypatch.setattr(
        monitor_mod,
        "monitor",
        AsyncMock(return_value=expected_report),
    )
    digest_calls = []
    monkeypatch.setattr(
        digest_mod,
        "send_monitoring_digest",
        lambda report, sd: digest_calls.append((report, sd)),
    )
    monkeypatch.setattr(type(config), "spend_dir", property(lambda self: tmp_path))

    monitor_mod.main()

    assert len(digest_calls) == 1
    assert digest_calls[0][0] is expected_report
    assert digest_calls[0][1] == tmp_path


def test_update_endpoints_main_calls_send_onboarding_digest(tmp_path, monkeypatch):
    """update_endpoints.main() must call send_onboarding_digest with the lifecycle report."""
    import asyncio

    import trackllm_website.bi.digest as digest_mod
    import trackllm_website.update_endpoints as ue
    from trackllm_website.config import config

    expected_report = OnboardingReport(
        date="2026-06-23",
        rows=[OnboardRow("m/x", "prov", "onboarded", 12, 0.02)],
    )

    monkeypatch.setattr(ue, "update_endpoints_lt", AsyncMock(return_value=None))
    monkeypatch.setattr(ue, "update_endpoints_bi", AsyncMock(return_value=[]))
    monkeypatch.setattr(
        ue, "update_endpoints_bi_lifecycle", AsyncMock(return_value=expected_report)
    )
    digest_calls = []
    monkeypatch.setattr(
        digest_mod,
        "send_onboarding_digest",
        lambda report, sd: digest_calls.append((report, sd)),
    )
    monkeypatch.setattr(type(config), "spend_dir", property(lambda self: tmp_path))

    asyncio.run(ue.main())

    assert len(digest_calls) == 1
    assert digest_calls[0][0] is expected_report
    assert digest_calls[0][1] == tmp_path
