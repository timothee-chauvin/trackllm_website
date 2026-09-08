// The readout's detail blocks: what a day, a change mark or an epoch rule on the
// endpoint chart is made of, spelled out from the raw votes and logprobs the series
// were computed from. Pure HTML builders. Prompts and tokens are provider output
// and go through esc, every one.
import { esc } from "./components";
import {
  type B3ITChange,
  type BICounts,
  type BIVote,
  type Counts,
  type FocusB3IT,
  type FocusLT,
  type LTChange,
  epochOf,
  fmtDrift,
  round,
} from "./chart_geom";

/** website/data/lt/<slug>/daily.json -- the schema is in generate_site/lt_raw.py. */
export interface LTRaw {
  drift_prompt: number;
  prompts: {
    text: string;
    tokens: string[];
    ref: (number | null)[]; // null: never returned in the reference period
    floor: number; // what drift scores such a token at
    days: [string, number, [number, number][]][];
  }[];
}

/** website/data/b3it/<slug>/votes.json (generate_site/b3it.py votes_json). */
export interface B3ITRaw {
  bis: string[]; // border-input prompts, indexed by BICounts
  reference: BICounts[][]; // per epoch, the ranked border inputs' reference votes
  batches: BIVote[][]; // per entry of FocusB3IT.daily, the votes behind it
}

/** Both raw files, fetched on first use: most readers never hover, and either
 *  file can outweigh the series many times over. */
export interface RawLoaders {
  lt: () => Promise<LTRaw | null>;
  b3it: () => Promise<B3ITRaw | null>;
}

// lt_drift.LT_SHIFT_WINDOW_DAYS: a level shift is the mean drift over this many
// days after the change against as many before it
export const LT_SHIFT_WINDOW_DAYS = 7;
const PROMPT_CHARS = 40;
const COUNT_TOKENS = 3; // votes listed per border input before the rest is counted

const clip = (s: string, n: number): string => (s.length > n ? `${s.slice(0, n - 1)}…` : s);
/** A token as the provider returned it, its whitespace made visible: quoted when
 *  it begins or ends with some, line breaks spelled out. */
const tok = (t: string): string => {
  const shown = t.replace(/\n/g, "⏎").replace(/\t/g, "⇥");
  return esc(shown === shown.trim() && shown ? shown : `"${shown}"`);
};
const promptCell = (text: string): string =>
  `<td class="p" title="${esc(text)}">${esc(clip(text, PROMPT_CHARS))}</td>`;
const mean = (vs: number[]): number => vs.reduce((a, b) => a + b, 0) / vs.length;
const block = (inner: string): string => `<div class="tip-detail">${inner}</div>`;
const humanize = (reason: string): string => reason.replace(/_/g, " ");

/** "Yes 31 · No 19 · +2". Sorted here: an object's integer-like keys ("1", "42")
 *  come out first whatever order they were written in. */
export function countsText(c: Counts): string {
  const entries = Object.entries(c).sort((x, y) => y[1] - x[1] || (x[0] < y[0] ? -1 : 1));
  const shown = entries.slice(0, COUNT_TOKENS).map(([t, n]) => `${tok(t)} ${n}`);
  if (entries.length > COUNT_TOKENS) shown.push(`+${entries.length - COUNT_TOKENS}`);
  return shown.join(" · ");
}

/** The day's mean logprobs per prompt, next to the reference period's for the
 *  same tokens: drift is the mean gap between the two columns. */
export function ltDayDetail(raw: LTRaw, date: string): string {
  const many = raw.prompts.length > 1;
  let floored = false;
  const sections = raw.prompts.map((p, i) => {
    const day = p.days.find((d) => d[0] === date);
    const who = `${tok(p.text)}${many && i === raw.drift_prompt ? " (the drift prompt)" : ""}`;
    if (!day) return `<div class="tip-h">${who} · no queries this day</div>`;
    const ref = (k: number): string => {
      const v = p.ref[k];
      if (v !== null) return v.toFixed(3);
      floored = true;
      return `<span class="dim">${p.floor.toFixed(3)}*</span>`;
    };
    const rows = day[2]
      .map(([k, v]) => `<tr><td>${tok(p.tokens[k])}</td><td class="n">${v.toFixed(3)}</td><td class="n">${ref(k)}</td></tr>`)
      .join("");
    return `<div class="tip-h">${who} · ${day[1]} queries</div><table class="tip-tbl"><tr><th>token</th><th class="n">day</th><th class="n">ref</th></tr>${rows}</table>`;
  });
  const note = floored
    ? `<p class="dim">* never returned in the reference period: scored at the series floor, as drift does.</p>`
    : "";
  return block(
    `<p>Mean logprob per top token (nats): this day vs the reference period.</p>${sections.join("")}${note}`
  );
}

/** Each border input's votes in the batch behind `b3it.daily[i]`, next to the
 *  epoch reference's -- more samples there, which is the point of showing counts. */
export function b3itDayDetail(b3it: FocusB3IT, raw: B3ITRaw, i: number): string {
  const [ts] = b3it.daily[i];
  const reference = raw.reference[epochOf(b3it.epochs, ts)];
  const votes = new Map(raw.batches[i].map(([k, day, tv]) => [k, { day, tv }]));
  const rows = reference
    .map(([k, ref]) => {
      const v = votes.get(k);
      return `<tr>${promptCell(raw.bis[k])}<td>${v ? countsText(v.day) : "—"}</td><td>${countsText(ref)}</td><td class="n">${v?.tv == null ? "—" : v.tv.toFixed(2)}</td></tr>`;
    })
    .join("");
  return block(
    `<p>Batch at ${esc(ts.slice(11, 16))} UTC: each border input's votes vs the epoch reference; the lane's TV is the mean of the last column.</p><table class="tip-tbl"><tr><th>prompt</th><th>day</th><th>reference</th><th class="n">TV</th></tr>${rows}</table>`
  );
}

export function ltChangeDetail(lt: FocusLT, c: LTChange): string {
  const before = lt.daily.filter(([d]) => d < c.date).slice(-LT_SHIFT_WINDOW_DAYS);
  const after = lt.daily.filter(([d]) => d >= c.date).slice(0, LT_SHIFT_WINDOW_DAYS);
  const level = (ps: [string, number][]): number => round(mean(ps.map(([, v]) => v)), 3);
  const how =
    before.length && after.length
      ? `mean drift ${level(after)} nats over the ${after.length} days after vs ${level(before)} over the ${before.length} days before.`
      : "its level could not be read off the plotted series.";
  return block(`<p><span class="badge lt">lt</span> Level shift ${fmtDrift(c.shift)}: ${how}</p>`);
}

const DETECTOR_COPY: Record<string, string> = {
  scan: " Called by the scan permutation test.",
  adaptive: " Called by the adaptive rule (consecutive deviating days).",
};

export function b3itChangeDetail(b3it: FocusB3IT, c: B3ITChange): string {
  const closed = b3it.epochs.findIndex((e) => e.changeDate?.slice(0, 10) === c.date);
  const ei = closed >= 0 ? closed : epochOf(b3it.epochs, `${c.date}T00:00:00Z`);
  const inEpoch = b3it.daily.filter(([ts]) => epochOf(b3it.epochs, ts) === ei);
  const level = (ps: [string, number][]): number => round(mean(ps.map(([, v]) => v)), 3);
  const before = inEpoch.filter(([ts]) => ts.slice(0, 10) < c.date);
  const after = inEpoch.filter(([ts]) => ts.slice(0, 10) >= c.date);
  const how = !after.length
    ? "its level could not be read off the plotted series."
    : before.length
      ? `mean TV ${level(after)} after vs ${level(before)} before the split, on the plotted series of epoch ${ei + 1}.`
      : `mean TV ${level(after)} over the first days of epoch ${ei + 1}, which has no days before the split.`;
  const shift = c.shiftTV === null ? "—" : `${round(c.shiftTV, 3)}`;
  return block(
    `<p><span class="badge b3it">b3it</span> TV shift ${shift}: ${how}${DETECTOR_COPY[c.detector ?? ""] ?? ""}</p>`
  );
}

/** Why the epoch began, and the reference votes every later day is measured
 *  against -- listed once `raw` has arrived, counted before. */
export function epochDetail(b3it: FocusB3IT, raw: B3ITRaw | null, i: number): string {
  const ep = b3it.epochs[i];
  const prev = b3it.epochs[i - 1];
  const after =
    prev?.endReason === "change_detected" && prev.changeDate
      ? `the change detected on ${esc(prev.changeDate.slice(0, 10))}`
      : `the previous epoch ended (${esc(humanize(prev?.endReason ?? "unknown"))})`;
  const why =
    i === 0
      ? "Initialised: border inputs searched and the reference sampled on this date. Each later day's TV is its distance from these votes."
      : `Re-initialised after ${after}. Border inputs and reference re-sampled: TV restarts from 0 against the new reference.`;
  const rows = (raw?.reference[i] ?? [])
    .map(([k, ref]) => `<tr>${promptCell(raw!.bis[k])}<td>${countsText(ref)}</td></tr>`)
    .join("");
  const table = ep.nRef
    ? `<div class="tip-h">Reference votes · ${ep.nRef} border inputs</div><table class="tip-tbl">${rows}</table>`
    : `<div class="tip-h">No reference was sampled.</div>`;
  return block(`<p>${why}</p>${table}`);
}
