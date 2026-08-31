"""Render and send the two daily B3IT digest emails (onboarding, monitoring)."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from html import escape

from trackllm_website import notify
from trackllm_website.bi.budget import month_to_date, projected_month_end
from trackllm_website.config import config, logger
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
    "gate_inconclusive": ("temperature gate inconclusive (retries)", "#bf8700"),
    "no_strategy": ("skipped: no strategy", "#6e7781"),
    "not_selected_budget": ("not selected: over budget", "#cf222e"),
    "probes_failed": ("all probes failed (not sampled)", "#cf222e"),
    "error": ("error (see logs)", "#cf222e"),
    "change_detected": ("change detected", "#0969da"),
    "reonboarded": ("re-onboarded", "#1a7f37"),
    "reonboard_no_bis": ("re-onboard → no BIs", "#cf222e"),
    "deadline_cutoff": ("cut off by the job deadline", "#cf222e"),
    "reinit_timeout": ("re-init timed out (retries tomorrow)", "#cf222e"),
    "retired_stalled": ("retired (stalled)", "#6e7781"),
    "retired_unreachable": ("retired (provider gone)", "#6e7781"),
    "retired_reinit_timeout": ("retired (re-init kept timing out)", "#6e7781"),
    "all_errors": ("all queries errored", "#cf222e"),
    "too_expensive": ("too expensive (guard tripped, cached)", "#cf222e"),
    "retired_too_expensive": ("retired (too expensive)", "#cf222e"),
    "skipped_budget": ("skipped (budget projection over cap)", "#cf222e"),
    "retired_budget": ("retired (budget)", "#cf222e"),
    "reinit_skipped_budget": ("re-init skipped (budget)", "#cf222e"),
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
    # endpoints where an exception escaped run_endpoint — a bug (API errors and
    # diagnosed conditions like re-init timeouts or unresolved probes are digest
    # rows, not failures). Non-empty fails the workflow after the digest is sent.
    failures: list[str] = field(default_factory=list)

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
    # model/provider are OpenRouter catalog strings, i.e. attacker-influenced input
    # rendered into our inbox. The href is safe by construction (slugify hex-encodes
    # everything outside its allowlist), but the link text must be escaped.
    return (
        f'<a href="{_url(model, provider)}" style="color:#0969da;text-decoration:none">'
        f"<b>{escape(model)}</b> "
        f'<span style="color:#6e7781">@ {escape(provider)}</span></a>'
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


def _budget_header(spend_dir, date):
    """(plain, html) month-end projection line for both digests: normal under
    target, amber between target and cap, red over cap."""
    now = datetime.fromisoformat(date).replace(tzinfo=timezone.utc)
    mtd = month_to_date(spend_dir, now)
    projected = projected_month_end(
        spend_dir, now, config.budget.projection_window_days
    )
    target = config.budget.target_per_month
    cap = config.budget.hard_cap_per_month
    base = (
        f"month-to-date {_money2(mtd)} · projected {_money2(projected)} "
        f"· target {_money2(target)} · cap {_money2(cap)}"
    )
    if projected > cap:
        status, color = "over cap", "#cf222e"
    elif projected > target:
        status, color = f"over target {_money2(target)}", "#bf8700"
    else:
        return f"Budget — {base}", f"<b>Budget</b> — {base}"
    return (
        f"Budget — {base} — {status}",
        f"<b>Budget</b> — {base} — "
        f'<span style="color:{color};font-weight:600">{status}</span>',
    )


def build_onboarding_email(report, spend_dir):
    tk = today_by_kind(spend_dir, report.date)
    ck = cumulative_by_kind(spend_dir)
    budget_plain, budget_html = _budget_header(spend_dir, report.date)
    onb_today = tk.get("onboard", 0) + tk.get("recheck", 0) + tk.get("vetting", 0)
    onb_cum = ck.get("onboard", 0) + ck.get("recheck", 0) + ck.get("vetting", 0)

    def c(o):
        return sum(1 for r in report.rows if r.outcome == o)

    # Budget skips must reach the headline: on a skip-only day, an all-zero
    # subject would bury the only red rows in the email.
    budget_note = f" · {n} over budget" if (n := c("not_selected_budget")) else ""
    # Projection-killer actions (skipped onboards + retired endpoints) are a
    # different signal from selection skips: count them separately.
    if killed := c("skipped_budget") + c("retired_budget"):
        budget_note += f" · {killed} budget-killed"
    summary = f"{c('onboarded')} onboarded · {c('timeout')} timed out · {c('no_bis')} not enough BIs{budget_note}"
    subject = f"[trackllm] {_money2(onb_today)} − B3IT onboarding: {c('onboarded')} onboarded, {c('timeout')} timed out, {c('no_bis')} no-BIs{budget_note}"
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
        f'<span style="color:#57606a">onboard {_money(tk.get("onboard", 0))} · rechecks {_money(tk.get("recheck", 0))} · vetting {_money(tk.get("vetting", 0))} (today)</span><br>'
        f"{budget_html}",
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
        + f"{budget_plain}\n"
    )
    return subject, plain, html


def build_monitoring_email(report, spend_dir):
    tk = today_by_kind(spend_dir, report.date)
    ck = cumulative_by_kind(spend_dir)
    budget_plain, budget_html = _budget_header(spend_dir, report.date)
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
        f" · re-init {_money(tk.get('reinit', 0))} (today)</span><br>"
        f"{budget_html}",
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
        + f"{budget_plain}\n"
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
