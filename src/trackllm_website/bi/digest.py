"""Render and send the two daily B3IT digest emails (onboarding, monitoring)."""

from dataclasses import dataclass

from trackllm_website import notify
from trackllm_website.config import logger
from trackllm_website.spend import cumulative_by_kind, today_by_kind
from trackllm_website.util import slugify

BASE_URL = "https://www.trackllm.net/endpoints"

OUTCOME = {
    "onboarded": ("onboarded", "#1a7f37"),
    "recheck_resurrected": ("recheck → resurrected", "#1a7f37"),
    "no_bis": ("not enough BIs", "#cf222e"),
    "recheck_still_no_bis": ("recheck → still no BIs", "#cf222e"),
    "timeout": ("timed out (resumes tomorrow)", "#bf8700"),
    "bad_temperature": ("ignores temperature (cached)", "#6e7781"),
    "no_strategy": ("skipped: no strategy", "#6e7781"),
    "error": ("error (see logs)", "#cf222e"),
    "change_detected": ("change detected", "#0969da"),
    "reonboarded": ("re-onboarded", "#1a7f37"),
    "reonboard_no_bis": ("re-onboard → no BIs", "#cf222e"),
    "retired_stalled": ("retired (stalled)", "#6e7781"),
}


@dataclass
class OnboardRow:
    model: str
    provider: str
    outcome: str
    n_bis: int | None
    spent: float


@dataclass
class OnboardingReport:
    date: str
    rows: list[OnboardRow]

    def notable(self) -> bool:
        return bool(self.rows)


@dataclass
class MonitorRow:
    model: str
    provider: str
    event: str
    change_date: str | None
    n_bis_after: int | None
    spent: float


@dataclass
class MonitorReport:
    date: str
    rows: list[MonitorRow]
    n_endpoints: int

    def notable(self) -> bool:
        return bool(self.rows)


def _money(x):
    return f"${x:,.4f}"


def _money2(x):
    return f"${x:,.2f}"


def _bis(n):
    return "—" if n is None else str(n)


def _url(model, provider):
    return f"{BASE_URL}/{slugify(f'{model}#{provider}')}.html"


def _label(key):
    return OUTCOME.get(key, (key, "#1f2328"))


def _link_html(model, provider):
    return (
        f'<a href="{_url(model, provider)}" style="color:#0969da;text-decoration:none">'
        f'<b>{model}</b> <span style="color:#6e7781">@ {provider}</span></a>'
    )


def _badge(key):
    lbl, color = _label(key)
    return f'<span style="color:{color};font-weight:600">{lbl}</span>'


def _table_html(headers, rows):
    th = "".join(
        f'<th style="text-align:left;padding:6px 12px;border-bottom:2px solid #d0d7de;font-size:13px;color:#57606a">{h}</th>'
        for h in headers
    )
    tr = "".join(
        "<tr>"
        + "".join(
            f'<td style="padding:6px 12px;border-bottom:1px solid #eaeef2;font-size:13px">{c}</td>'
            for c in r
        )
        + "</tr>"
        for r in rows
    )
    return f'<table style="border-collapse:collapse;width:100%;font-family:ui-monospace,Menlo,monospace"><tr>{th}</tr>{tr}</table>'


def _shell(title, summary, table, footer):
    return (
        f'<div style="font-family:-apple-system,Segoe UI,Helvetica,Arial,sans-serif;color:#1f2328;max-width:760px;margin:0 auto">'
        f'<h2 style="margin:0 0 2px">{title}</h2>'
        f'<div style="color:#57606a;font-size:14px;margin-bottom:16px">{summary}</div>{table}'
        f'<div style="margin-top:18px;padding:12px 14px;background:#f6f8fa;border:1px solid #d0d7de;border-radius:6px;font-size:13px">{footer}</div></div>'
    )


def _plain_table(headers, rows, widths):
    def line(cells):
        return "  ".join(str(c).ljust(w) for c, w in zip(cells, widths))

    return "\n".join(
        [line(headers), line(["-" * w for w in widths])] + [line(r) for r in rows]
    )


def build_onboarding_email(report, spend_dir):
    tk = today_by_kind(spend_dir, report.date)
    ck = cumulative_by_kind(spend_dir)
    onb_today = tk.get("onboard", 0) + tk.get("recheck", 0) + tk.get("vetting", 0)
    onb_cum = ck.get("onboard", 0) + ck.get("recheck", 0) + ck.get("vetting", 0)

    def c(o):
        return sum(1 for r in report.rows if r.outcome == o)

    summary = f"{c('onboarded')} onboarded · {c('timeout')} timed out · {c('no_bis')} not enough BIs"
    subject = f"[trackllm] {_money2(onb_today)} − B3IT onboarding: {c('onboarded')} onboarded, {c('timeout')} timed out, {c('no_bis')} no-BIs"
    hrows = [
        (
            _link_html(r.model, r.provider),
            _badge(r.outcome),
            _bis(r.n_bis),
            _money(r.spent),
        )
        for r in report.rows
    ]
    html = _shell(
        f"B3IT onboarding — {report.date}",
        summary,
        _table_html(["Endpoint", "Outcome", "BIs", "Spent today"], hrows),
        f"<b>Onboarding-run spend</b> — today <b>{_money2(onb_today)}</b> · cumulative {_money(onb_cum)}<br>"
        f'<span style="color:#57606a">onboard {_money(tk.get("onboard", 0))} · rechecks {_money(tk.get("recheck", 0))} · vetting {_money(tk.get("vetting", 0))} (today)</span>',
    )
    prows = [
        (
            f"{r.model} @ {r.provider}",
            _label(r.outcome)[0],
            _bis(r.n_bis),
            _money(r.spent),
        )
        for r in report.rows
    ]
    plain = (
        f"B3IT onboarding — {report.date}\n{summary}\n\n"
        + _plain_table(
            ["Endpoint", "Outcome", "BIs", "Spent today"], prows, [46, 30, 4, 10]
        )
        + f"\n\nOnboarding-run spend — today {_money2(onb_today)} · cumulative {_money(onb_cum)}\n"
    )
    return subject, plain, html


def build_monitoring_email(report, spend_dir):
    tk = today_by_kind(spend_dir, report.date)
    ck = cumulative_by_kind(spend_dir)
    mon_today = tk.get("monitor", 0) + tk.get("reinit", 0)
    mon_cum = ck.get("monitor", 0) + ck.get("reinit", 0)
    n_changes = sum(
        1
        for r in report.rows
        if r.event in ("change_detected", "reonboarded", "reonboard_no_bis")
    )
    summary = f"{n_changes} changes detected across {report.n_endpoints} endpoints"
    subject = f"[trackllm] {_money2(mon_today)} − B3IT monitoring: {n_changes} changes detected"
    hrows = [
        (
            _link_html(r.model, r.provider),
            _badge(r.event),
            r.change_date or "—",
            _bis(r.n_bis_after),
            _money(r.spent),
        )
        for r in report.rows
    ]
    html = _shell(
        f"B3IT monitoring — {report.date}",
        summary,
        _table_html(
            ["Endpoint", "Event", "Change date", "BIs after", "Re-onboard $"], hrows
        ),
        f"<b>Monitoring-run spend</b> — today <b>{_money2(mon_today)}</b> · cumulative {_money(mon_cum)}<br>"
        f'<span style="color:#57606a">monitoring {_money(tk.get("monitor", 0))} across {report.n_endpoints} endpoints'
        f" · re-init {_money(tk.get('reinit', 0))} (today)</span>",
    )
    prows = [
        (
            f"{r.model} @ {r.provider}",
            _label(r.event)[0],
            r.change_date or "—",
            _bis(r.n_bis_after),
            _money(r.spent),
        )
        for r in report.rows
    ]
    plain = (
        f"B3IT monitoring — {report.date}\n{summary}\n\n"
        + _plain_table(
            ["Endpoint", "Event", "Change date", "BIs after", "Re-onboard $"],
            prows,
            [46, 20, 12, 9, 10],
        )
        + f"\n\nMonitoring-run spend — today {_money2(mon_today)} · cumulative {_money(mon_cum)}\n"
    )
    return subject, plain, html


def send_onboarding_digest(report, spend_dir):
    if not report.notable():
        logger.info("onboarding digest: nothing notable, skipping")
        return
    subject, plain, html = build_onboarding_email(report, spend_dir)
    notify.send_email(notify.load_creds_from_env(), subject, plain, html)


def send_monitoring_digest(report, spend_dir):
    if not report.notable():
        logger.info("monitoring digest: nothing notable, skipping")
        return
    subject, plain, html = build_monitoring_email(report, spend_dir)
    notify.send_email(notify.load_creds_from_env(), subject, plain, html)
