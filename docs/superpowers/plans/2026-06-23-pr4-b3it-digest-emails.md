# PR4 — B3IT Daily Digest Emails Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Two daily B3IT emails (not LT) — one at the end of the `update-endpoints` run (onboarding), one at the end of the `bi-monitor` run (monitoring) — built in-process from run outcomes + the spend ledger (PR3), sent only when something notable happened.

**Architecture:** Each run builds an in-memory report (per-endpoint outcome rows) and hands it to `bi/digest.py`, which renders an HTML + plain-text email (subject leads with the run's `$` cost; endpoint names link to trackllm.net; per-kind spend footer with today + cumulative read from the ledger) and sends via a reusable stdlib `send_email` refactored out of `notify.py`. Gated: skip when the report has no rows.

**Tech Stack:** Python 3.13 (stdlib `email`/`smtplib`), pydantic/dataclasses, orjson, pytest, uv; GitHub Actions.

## Global Constraints (locked email format — validated with the user via mockups)

- **Subjects:** `[trackllm] $X.XX − B3IT onboarding: 4 onboarded, 1 timed out, 2 no-BIs` and `[trackllm] $X.XX − B3IT monitoring: 2 changes detected`. The `$X.XX` is the **whole run's** B3IT spend (2 decimals) so the two subjects sum to the day's total. Dash char is `−` (U+2212).
- **Onboarding-run `$`** = today's `onboard + recheck + vetting` (ledger). **Monitoring-run `$`** = today's `monitor + reinit`.
- **Endpoint cell links** to `https://www.trackllm.net/endpoints/{slugify(f"{model}#{provider}")}.html` (use `trackllm_website.util.slugify`).
- **HTML** (color-coded outcome labels, monospace table) **with a plain-text alternative**. Footer = per-kind today + cumulative spend.
- **Gate:** send only if the report has ≥1 row. Quiet days → no email (spend still in the ledger; next email shows cumulative).
- `money(x)=f"${x:,.4f}"` in bodies; `money2(x)=f"${x:,.2f}"` in subjects.
- B3IT only — never email about LT.
- `notify.py` must stay pure-stdlib (the failure watcher runs on system Python before `uv sync`).
- `prek run --files <changed .py>` before each commit; `git commit --no-verify`. Tests: `OPENROUTER_API_KEY=dummy uv run pytest`.

---

### Task 1: Refactor `notify.py` to expose a reusable `send_email`

**Files:** Modify `src/trackllm_website/notify.py`; Test `tests/test_notify.py` (exists — extend).

**Interfaces produced:** `send_email(creds: dict[str, str], subject: str, plain: str, html: str | None = None) -> None` (SMTP_SSL to Gmail; `add_alternative(html, subtype="html")` when html given). `load_creds_from_env()` stays. The existing failure-watcher `main()` must keep working (build subject/body, call `send_email`).

- [ ] **Step 1: Write the failing test** — in `tests/test_notify.py`, monkeypatch `smtplib.SMTP_SSL` with a fake recording the sent `EmailMessage`; assert `send_email(creds, "subj", "plain", "<b>h</b>")` produces a message with `Subject==subj`, a `text/plain` part == "plain", and a `text/html` alternative containing "<b>h</b>"; and that `html=None` yields a plain-only message.

```python
def test_send_email_multipart(monkeypatch):
    sent = {}
    class FakeSMTP:
        def __init__(self, *a, **k): pass
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def login(self, *a): pass
        def send_message(self, msg): sent["msg"] = msg
    monkeypatch.setattr("trackllm_website.notify.smtplib.SMTP_SSL", FakeSMTP)
    from trackllm_website.notify import send_email
    send_email({"GMAIL_USER": "u@x", "GMAIL_APP_PASSWORD": "p", "NOTIFY_EMAIL": "to@x"},
               "subj", "the plain text", "<b>hello</b>")
    msg = sent["msg"]
    assert msg["Subject"] == "subj" and msg["To"] == "to@x"
    body = msg.get_body(("html",))
    assert body is not None and "hello" in body.get_content()
```

- [ ] **Step 2: Run → FAIL** (`send_email` not importable): `OPENROUTER_API_KEY=dummy uv run pytest tests/test_notify.py -q`
- [ ] **Step 3: Implement.** Extract the SMTP send into:

```python
def send_email(creds: dict[str, str], subject: str, plain: str, html: str | None = None) -> None:
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
```

Rewrite the failure-watcher `main()` to build `subject, body` as today and call `send_email(creds, subject, body)`. Keep `build_message`, `load_creds_from_env`, the `REQUIRED_KEYS`, imports stdlib-only.

- [ ] **Step 4: Run → PASS** (new test + existing notify tests).
- [ ] **Step 5: Commit** (`feat(notify): reusable send_email(creds, subject, plain, html=None)`).

---

### Task 2: `bi/digest.py` — report types, ledger today-totals, rendering (the locked format)

**Files:** Create `src/trackllm_website/bi/digest.py`; Modify `src/trackllm_website/spend.py` (add `today_by_kind`); Test `tests/test_digest.py` (create).

**Interfaces produced (consumed by Tasks 3–5):**
- `spend.today_by_kind(spend_dir: Path, day: str) -> dict[str, float]` — like `cumulative_by_kind` but only entries whose `timestamp` starts with `day` (`YYYY-MM-DD`).
- `@dataclass OnboardRow(model, provider, outcome: str, n_bis: int | None, spent: float)`
- `@dataclass OnboardingReport(date: str, rows: list[OnboardRow])` with `notable() -> bool` (= `bool(rows)`).
- `@dataclass MonitorRow(model, provider, event: str, change_date: str | None, n_bis_after: int | None, spent: float)`
- `@dataclass MonitorReport(date: str, rows: list[MonitorRow], n_endpoints: int)` with `notable()`.
- `build_onboarding_email(report, spend_dir) -> tuple[str, str, str]` → `(subject, plain, html)`.
- `build_monitoring_email(report, spend_dir) -> tuple[str, str, str]`.
- `send_onboarding_digest(report, spend_dir) -> None` and `send_monitoring_digest(report, spend_dir) -> None` — gate on `report.notable()`; else log `"digest: nothing notable, skipping"` and return; otherwise build + `notify.send_email(notify.load_creds_from_env(), subject, plain, html)`.

- [ ] **Step 1: Write failing tests** (`tests/test_digest.py`): build a sample `OnboardingReport` with a mix of outcomes and a tmp ledger (write `onboard`/`recheck`/`vetting` lines for today + an older month via `spend.append_entry`); assert: subject startswith `"[trackllm] $"`, contains `"− B3IT onboarding:"` and the today total (2dp); plain contains an outcome label; html contains a `trackllm.net/endpoints/` link with the right slug. Same for monitoring. Plus: `notable()` false on empty rows → `send_*_digest` does NOT call `send_email` (monkeypatch `notify.send_email`). Plus a `today_by_kind` unit test (today vs older entries).

- [ ] **Step 2: Run → FAIL.**
- [ ] **Step 3: Implement.** `spend.today_by_kind`:

```python
def today_by_kind(spend_dir: Path, day: str) -> dict[str, float]:
    totals: dict[str, float] = defaultdict(float)
    if not spend_dir.exists():
        return dict(totals)
    for f in spend_dir.glob("*/*.jsonl"):
        for line in f.read_bytes().splitlines():
            if not line.strip():
                continue
            rec = orjson.loads(line)
            if str(rec["timestamp"]).startswith(day):
                totals[rec["kind"]] += rec["cost"]
    return dict(totals)
```

`bi/digest.py` (reproduce the approved mockup format):

```python
"""Render and send the two daily B3IT digest emails (onboarding, monitoring)."""

from dataclasses import dataclass, field
from pathlib import Path

from trackllm_website import notify
from trackllm_website.config import logger
from trackllm_website.spend import cumulative_by_kind, today_by_kind
from trackllm_website.util import slugify

BASE_URL = "https://www.trackllm.net/endpoints"

OUTCOME = {  # key -> (label, color)
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

def _money(x): return f"${x:,.4f}"
def _money2(x): return f"${x:,.2f}"
def _bis(n): return "—" if n is None else str(n)
def _url(model, provider): return f"{BASE_URL}/{slugify(f'{model}#{provider}')}.html"
def _label(key): return OUTCOME.get(key, (key, "#1f2328"))

def _link_html(model, provider):
    return (f'<a href="{_url(model, provider)}" style="color:#0969da;text-decoration:none">'
            f'<b>{model}</b> <span style="color:#6e7781">@ {provider}</span></a>')

def _badge(key):
    lbl, color = _label(key)
    return f'<span style="color:{color};font-weight:600">{lbl}</span>'

def _table_html(headers, rows):
    th = "".join(f'<th style="text-align:left;padding:6px 12px;border-bottom:2px solid #d0d7de;font-size:13px;color:#57606a">{h}</th>' for h in headers)
    tr = "".join("<tr>" + "".join(f'<td style="padding:6px 12px;border-bottom:1px solid #eaeef2;font-size:13px">{c}</td>' for c in r) + "</tr>" for r in rows)
    return f'<table style="border-collapse:collapse;width:100%;font-family:ui-monospace,Menlo,monospace"><tr>{th}</tr>{tr}</table>'

def _shell(title, summary, table, footer):
    return (f'<div style="font-family:-apple-system,Segoe UI,Helvetica,Arial,sans-serif;color:#1f2328;max-width:760px;margin:0 auto">'
            f'<h2 style="margin:0 0 2px">{title}</h2>'
            f'<div style="color:#57606a;font-size:14px;margin-bottom:16px">{summary}</div>{table}'
            f'<div style="margin-top:18px;padding:12px 14px;background:#f6f8fa;border:1px solid #d0d7de;border-radius:6px;font-size:13px">{footer}</div></div>')

def _plain_table(headers, rows, widths):
    line = lambda cells: "  ".join(str(c).ljust(w) for c, w in zip(cells, widths))
    return "\n".join([line(headers), line(["-" * w for w in widths])] + [line(r) for r in rows])
```

Onboarding builder (subject/summary/footer use today/cumulative ledger reads):

```python
def build_onboarding_email(report, spend_dir):
    tk = today_by_kind(spend_dir, report.date); ck = cumulative_by_kind(spend_dir)
    onb_today = tk.get("onboard", 0) + tk.get("recheck", 0) + tk.get("vetting", 0)
    onb_cum = ck.get("onboard", 0) + ck.get("recheck", 0) + ck.get("vetting", 0)
    c = lambda o: sum(1 for r in report.rows if r.outcome == o)
    summary = f"{c('onboarded')} onboarded · {c('timeout')} timed out · {c('no_bis')} not enough BIs"
    subject = f"[trackllm] {_money2(onb_today)} − B3IT onboarding: {c('onboarded')} onboarded, {c('timeout')} timed out, {c('no_bis')} no-BIs"
    hrows = [(_link_html(r.model, r.provider), _badge(r.outcome), _bis(r.n_bis), _money(r.spent)) for r in report.rows]
    html = _shell(f"B3IT onboarding — {report.date}", summary,
                  _table_html(["Endpoint", "Outcome", "BIs", "Spent today"], hrows),
                  f"<b>Onboarding-run spend</b> — today <b>{_money2(onb_today)}</b> · cumulative {_money(onb_cum)}<br>"
                  f'<span style="color:#57606a">onboard {_money(tk.get("onboard",0))} · rechecks {_money(tk.get("recheck",0))} · vetting {_money(tk.get("vetting",0))} (today)</span>')
    prows = [(f"{r.model} @ {r.provider}", _label(r.outcome)[0], _bis(r.n_bis), _money(r.spent)) for r in report.rows]
    plain = (f"B3IT onboarding — {report.date}\n{summary}\n\n"
             + _plain_table(["Endpoint", "Outcome", "BIs", "Spent today"], prows, [46, 30, 4, 10])
             + f"\n\nOnboarding-run spend — today {_money2(onb_today)} · cumulative {_money(onb_cum)}\n")
    return subject, plain, html
```

Monitoring builder is analogous: `mon_today = tk.get("monitor",0)+tk.get("reinit",0)`; subject `… − B3IT monitoring: {n_changes} changes detected` where `n_changes = sum(r.event in ("change_detected","reonboarded","reonboard_no_bis") ...)`; columns `["Endpoint","Event","Change date","BIs after","Re-onboard $"]`; footer shows monitoring `{tk[monitor]}` across `report.n_endpoints` endpoints + re-init `{tk[reinit]}`, today + cumulative. (Implement symmetrically to the onboarding builder above.)

Senders:

```python
def send_onboarding_digest(report, spend_dir):
    if not report.notable():
        logger.info("onboarding digest: nothing notable, skipping"); return
    subject, plain, html = build_onboarding_email(report, spend_dir)
    notify.send_email(notify.load_creds_from_env(), subject, plain, html)

def send_monitoring_digest(report, spend_dir):
    if not report.notable():
        logger.info("monitoring digest: nothing notable, skipping"); return
    subject, plain, html = build_monitoring_email(report, spend_dir)
    notify.send_email(notify.load_creds_from_env(), subject, plain, html)
```

- [ ] **Step 4: Run → PASS.**
- [ ] **Step 5: Commit** (`feat(digest): bi/digest.py rendering + spend.today_by_kind`).

---

### Task 3: Build the `OnboardingReport` in the lifecycle run

**Files:** Modify `src/trackllm_website/update_endpoints.py`; Test `tests/test_digest_onboarding_report.py` (create).

**Interfaces:** `update_endpoints_bi_lifecycle(candidates) -> OnboardingReport` (currently returns `None`).

- [ ] **Step 1: Write failing tests** — mirror `tests/test_spend_lifecycle.py` stubbing; drive the lifecycle with a fake `reinit` returning an epoch (→ `onboarded`, `n_bis == len(border_inputs)`) and one returning `epoch=None` (→ `no_bis`); assert the returned `OnboardingReport.rows` contain those outcomes with the right `n_bis`. Add a recheck case (`is_recheck=True` → `recheck_resurrected`/`recheck_still_no_bis`). MUST patch `config.spend_dir` to tmp (lesson: tests pollute the repo otherwise).

- [ ] **Step 2: Run → FAIL.**
- [ ] **Step 3: Implement.** In `update_endpoints_bi_lifecycle`, create `report_rows: list[OnboardRow] = []` before the gather. In `onboard_one`, track `outcome` (default `"error"`) and `n_bis` (default `None`), set them at each branch:
  - strategy-unresolved skip → `outcome = "no_strategy"`
  - `bad_temperature` → `"bad_temperature"`
  - `result.epoch is None` → `"recheck_still_no_bis" if is_recheck else "no_bis"`
  - epoch set → `n_bis = len(result.epoch.border_inputs)`; `"recheck_resurrected" if is_recheck else "onboarded"`
  - `except asyncio.TimeoutError` → `"timeout"`; `except Exception` → `"error"`
  In the existing `finally` (after `append_entry`), append `OnboardRow(endpoint.model, endpoint.provider, outcome, n_bis, spend.cost)` to `report_rows`. Return `OnboardingReport(date=now.date().isoformat(), rows=report_rows)` at the end — and also from the early `if not to_init: return` path (return `OnboardingReport(now.date().isoformat(), [])`). `import` the dataclasses from `bi.digest`.

- [ ] **Step 4: Run → PASS**, regression `tests/test_bi_lifecycle.py tests/test_spend_lifecycle.py -q`.
- [ ] **Step 5: Commit** (`feat(digest): build OnboardingReport from the lifecycle run`).

---

### Task 4: Build the `MonitorReport` in the monitor run

**Files:** Modify `src/trackllm_website/bi/monitor.py`; Test `tests/test_digest_monitor_report.py` (create).

**Interfaces:** `monitor() -> MonitorReport` (currently `None`); `run_endpoint(..., event_rows: list[MonitorRow] | None = None)` appends an event row on reinit/retire.

- [ ] **Step 1: Write failing tests** — mirror `tests/test_spend_monitor.py`; force `decide()` → `reinit` (assert a `reonboarded` row with `change_date` + `n_bis_after`) and → `retire_stalled` (assert `retired_stalled` row); an `action="none"` run adds NO row. Patch `config.spend_dir` to tmp.

- [ ] **Step 2: Run → FAIL.**
- [ ] **Step 3: Implement.** Add `event_rows` param to `run_endpoint`; when `decision.action == "retire_stalled"` append `MonitorRow(model, provider, "retired_stalled", None, None, 0.0)`; on `reinit`, after the reinit block, append `MonitorRow(model, provider, event=("reonboarded" if result.epoch else "reonboard_no_bis"), change_date=decision.change_date.date().isoformat() if decision.change_date else None, n_bis_after=(len(result.epoch.border_inputs) if result.epoch else None), spent=reinit_spend.cost)`. (Use `"change_detected"` only if you prefer to also emit a row when a change is detected but reinit is skipped — here reinit always runs, so `reonboarded`/`reonboard_no_bis` cover it.) In `monitor()`, create `event_rows = []`, thread it into `run_isolated`/`run_endpoint`, and `return MonitorReport(date=now.date().isoformat(), rows=event_rows, n_endpoints=len(monitoring))`.

- [ ] **Step 4: Run → PASS**, regression `tests/test_bi_monitor.py tests/test_spend_monitor.py -q`.
- [ ] **Step 5: Commit** (`feat(digest): build MonitorReport from the monitor run`).

---

### Task 5: Wire senders into the entrypoints + add workflow secrets

**Files:** Modify `src/trackllm_website/update_endpoints.py` (`main`), `src/trackllm_website/bi/monitor.py` (`monitor`/`__main__`), `.github/workflows/update-endpoints.yml`, `.github/workflows/bi-monitor.yml`; Test `tests/test_digest_wiring.py` (create).

- [ ] **Step 1: Write failing tests** — monkeypatch `bi.digest.send_onboarding_digest` / `send_monitoring_digest` (or `notify.send_email`) and assert: with a report having rows it IS called once; with an empty report it is NOT called. (Drive via the entrypoints with the same stubbing as Tasks 3–4, or call the senders directly with empty vs non-empty reports — the gating test in Task 2 already covers the latter; here assert the entrypoint passes its report through.)

- [ ] **Step 2: Run → FAIL.**
- [ ] **Step 3: Implement.**
  - `update_endpoints.py::main`: `report = await update_endpoints_bi_lifecycle(good)`; then `from trackllm_website.bi.digest import send_onboarding_digest; send_onboarding_digest(report, config.spend_dir)`.
  - `bi/monitor.py`: have `monitor()` return the report; in the `fire.Fire(monitor)`/`__main__` wrapper (or at the end of `monitor()`), call `send_monitoring_digest(report, config.spend_dir)`. Keep `monitor()` returning the report for testability; do the send in a thin `def main(): report = asyncio.run(monitor()); send_monitoring_digest(report, config.spend_dir)` and point `fire.Fire(main)`.
  - Both `update-endpoints.yml` and `bi-monitor.yml`: add to the python run step's `env:` block, alongside `OPENROUTER_API_KEY`:
    ```yaml
        GMAIL_USER: ${{ secrets.GMAIL_USER }}
        GMAIL_APP_PASSWORD: ${{ secrets.GMAIL_APP_PASSWORD }}
        NOTIFY_EMAIL: ${{ secrets.NOTIFY_EMAIL }}
    ```
  A send failure is allowed to raise (→ existing failure-email path); the run's data is already committed before the send.

- [ ] **Step 4: Run → PASS**, then full suite `OPENROUTER_API_KEY=dummy uv run pytest -q`.
- [ ] **Step 5: Commit** (`feat(digest): send onboarding/monitoring digests from the daily runs + workflow secrets`).

---

## Self-Review

**Spec coverage:** reusable `send_email` (T1); rendering + ledger today/cumulative + gating, locked format incl. cost-subject/links/footer (T2); onboarding report with outcomes+BIs+spend (T3); monitor report with events (T4); in-process send + gate + workflow secrets (T5). ✓
**Placeholder scan:** Task 2's monitoring builder and Task 5's tests say "analogous/symmetric to" the fully-shown onboarding builder — deliberate (the onboarding builder is shown complete; the monitoring one mirrors it with the stated columns/kinds). All new prod interfaces have complete code or exact field/signature specs.
**Type consistency:** `OnboardRow`/`OnboardingReport`/`MonitorRow`/`MonitorReport`, `today_by_kind(spend_dir, day)`, `cumulative_by_kind`, `build_*_email -> (subject, plain, html)`, `send_*_digest(report, spend_dir)`, `notify.send_email(creds, subject, plain, html=None)` — names/signatures identical across T1–T5. Tests must patch `config.spend_dir` to tmp (repo-pollution lesson).
**Out of scope:** the carried-over failed-probe ledger Minor (separate follow-up; digest reads whatever the ledger has).
