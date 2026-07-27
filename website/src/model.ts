// `export {}` makes this a module so its top-level names (MONTHS, td, init, ...)
// don't collide with the same names in other bundler-entrypoint scripts (endpoint.ts)
// when type-checked together as one tsc program.
export {};

interface LTChange {
  date: string;
  sigma: string;
  drift: number;
}

interface B3ITChange {
  date: string;
  peakTV: number;
}

interface EndpointLT {
  drift: [string, number][];
  changes: LTChange[];
}

interface EndpointB3IT {
  tv: [string, number][];
  changes: B3ITChange[];
}

interface ModelEndpoint {
  slug: string;
  provider: string;
  methods: string[];
  first: string | null;
  last: string | null;
  n_changes: number;
  lt: EndpointLT | null;
  b3it: EndpointB3IT | null;
}

interface ModelData {
  model: string;
  org: string;
  date_min: string | null;
  date_max: string | null;
  n_providers: number;
  n_changed: number;
  max_drift: number;
  endpoints: ModelEndpoint[];
}

const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
const td = (s: string): number => Date.parse(s + "T00:00:00Z");
const DAY_MS = 86400_000;

async function init(): Promise<void> {
  const cmpEl = document.getElementById("cmp");
  const slugEl = document.getElementById("model-slug");
  if (!cmpEl || !slugEl) return;

  const slug: string = JSON.parse(slugEl.textContent || '""');
  const res = await fetch(`../data/models/${slug}.json`).catch(() => null);
  const D: ModelData | null = res && res.ok ? await res.json() : null;

  if (!D || !D.date_min || !D.date_max || !D.endpoints.length) {
    cmpEl.innerHTML = `<div style="padding:2rem 1rem;color:var(--text-dim);font-size:0.85rem">No monitoring data available yet for this model.</div>`;
    return;
  }

  const D0 = td(D.date_min);
  const D1 = td(D.date_max);
  const W = 1000;
  const xpos = (s: string): number => ((td(s) - D0) / (D1 - D0)) * W;

  const ledeEl = document.getElementById("lede");
  if (ledeEl) {
    ledeEl.innerHTML = `Served by <b>${D.n_providers}</b> providers. <span class="hl">${D.n_changed}</span> of them show at least one detected change since launch — evidence the served behaviour drifts even when the model version doesn't.`;
  }
  const summaryEl = document.getElementById("summary");
  if (summaryEl) {
    summaryEl.innerHTML = `
      <div class="s"><div class="v">${D.n_providers}</div><div class="k">Providers</div></div>
      <div class="s"><div class="v" style="color:var(--changed)">${D.n_changed}</div><div class="k">With changes</div></div>
      <div class="s"><div class="v">${MONTHS[new Date(D0).getUTCMonth()]} ${new Date(D0).getUTCFullYear()} – ${MONTHS[new Date(D1).getUTCMonth()]} ${new Date(D1).getUTCFullYear()}</div><div class="k">Monitored</div></div>`;
  }

  function monthTicks(): Date[] {
    const out: Date[] = [];
    const d = new Date(D0);
    d.setUTCDate(1);
    while (d.getTime() <= D1) {
      if (d.getTime() >= D0 - 15 * DAY_MS) out.push(new Date(d));
      d.setUTCMonth(d.getUTCMonth() + 1);
    }
    return out;
  }

  // shared per-method scales so strip heights are directly comparable
  const LT_MAX = Math.max(
    0.5,
    ...D.endpoints.filter((e) => e.lt).flatMap((e) => e.lt!.drift.map((p) => p[1]))
  );

  function strip(ep: ModelEndpoint): string {
    const sig = ep.lt ? ep.lt.drift : ep.b3it ? ep.b3it.tv : [];
    const dmax = ep.lt ? LT_MAX : 1;
    const H = 40, pad = 5;
    let path = "";
    if (sig.length > 1) {
      path = sig
        .map((p, i) => {
          const v = Math.min(p[1], dmax);
          return `${i ? "L" : "M"}${xpos(p[0]).toFixed(1)} ${(H - pad - (v / dmax) * (H - 2 * pad)).toFixed(1)}`;
        })
        .join(" ");
    }
    const grid = monthTicks()
      .map((d) => {
        const x = xpos(d.toISOString().slice(0, 10));
        return `<line x1="${x.toFixed(1)}" y1="2" x2="${x.toFixed(1)}" y2="${H - 2}" stroke="var(--border-soft)" stroke-width="1"/>`;
      })
      .join("");
    const marks: string[] = [];
    (ep.lt ? ep.lt.changes : []).forEach((c) =>
      marks.push(
        `<circle cx="${xpos(c.date).toFixed(1)}" cy="${H / 2}" r="3.6" fill="var(--accent)"><title>LT change ${c.date} · ${c.sigma}, drift ${c.drift}</title></circle>`
      )
    );
    (ep.b3it ? ep.b3it.changes : []).forEach((c) =>
      marks.push(
        `<circle cx="${xpos(c.date).toFixed(1)}" cy="${H / 2}" r="3.6" fill="var(--b3it)"><title>B3IT change ${c.date} · TV ${c.peakTV}</title></circle>`
      )
    );
    const col = ep.lt ? "var(--accent)" : "var(--b3it)";
    return `<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none">${grid}
      ${path ? `<path d="${path}" fill="none" stroke="${col}" stroke-width="1.3" opacity="0.5" vector-effect="non-scaling-stroke"/>` : ""}${marks.join("")}</svg>`;
  }

  cmpEl.innerHTML =
    D.endpoints
      .map((ep) => {
        const peak = ep.lt
          ? `${Math.max(...ep.lt.drift.map((p) => p[1])).toFixed(1)} nats`
          : ep.b3it
            ? `TV ${Math.max(...ep.b3it.tv.map((p) => p[1])).toFixed(2)}`
            : "—";
        const n = ep.n_changes;
        return `<div class="row">
      <div class="pv"><a href="../endpoints/${ep.slug}.html">${ep.provider}</a><div class="mm">${ep.methods.map((m) => `<span class="badge ${m}">${m}</span>`).join("")}</div></div>
      <div class="spark">${strip(ep)}</div>
      <div class="meta"><div class="chg"><span class="${n ? "some" : "zero"}">${n || "—"} chg</span></div><div class="peak">${peak}</div></div>
    </div>`;
      })
      .join("") + `<div class="axis"><div class="ticks" id="ticks"></div></div>`;

  const ticksEl = document.getElementById("ticks");
  if (ticksEl) {
    ticksEl.innerHTML = monthTicks()
      .filter((_, i) => i % 2 === 0)
      .map(
        (d) =>
          `<span style="left:${((xpos(d.toISOString().slice(0, 10)) / W) * 100).toFixed(2)}%">${MONTHS[d.getUTCMonth()]} ${String(d.getUTCFullYear()).slice(2)}</span>`
      )
      .join("");
  }
}

init();
