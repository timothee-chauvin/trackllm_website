// data/overview.json is the one file the Overview, Providers and Endpoints pages
// all render from, so its shape and its loading live here.
import { FeedItem, showLoadError } from "./components";
import { EndpointRow } from "./directory";

export interface Stats {
  // headline counts, so they agree with the directory's status chips: `endpoints`
  // is the fleet we ever tracked (tracked + retired) and `active` its tracked half
  endpoints: number;
  catalog_endpoints: number; // every row of the directory, series or not
  providers: number;
  provider_companies: number;
  models: number;
  orgs: number;
  changes_total: number;
  changes_lt: number;
  changes_b3it: number;
  active: number;
  changed_endpoints: number;
  changes_30d: number;
  lt_endpoints: number;
  b3it_endpoints: number;
  b3it_monitoring: number;
  b3it_since: string | null;
  queries: number;
  since: string | null;
  spend_cumulative: number;
  now: string | null;
  // absolute instants (overview.py), turned into an age here at page load
  last_query_lt: string | null;
  last_query_b3it: string | null;
}

/** One provider *company*, with its serving variants pooled (provider.py::overview_rows).
 *  `lt_rate` and `lt_ci` are null together, and that null is the "not enough
 *  monitoring" state — never a rate of zero, and never recomputed here. */
export interface ProviderRate {
  name: string;
  slug: string;
  brand: Brand;
  n_endpoints: number;
  n_models: number;
  n_variants: number;
  lt_years: number;
  lt_changes: number;
  lt_rate: number | null;
  lt_ci: [number, number] | null;
  b3it_endpoints: number;
  b3it_years: number;
  last_change: string | null;
}

/** How a provider is shown (brands.py / provider_brands.yaml): its display name
 *  and, when one was found, its logo. A wordmark logo *is* the name. */
export interface Brand {
  name: string;
  logo: string | null;
  kind: "icon" | "wordmark";
  mono: boolean; // a flat dark mark: inverted in the dark theme
  dark: string | null; // a light-on-dark variant, shown in the dark theme instead
}

/** The one real change event the hero draws, chosen at build time (hero.py). */
export interface Hero {
  slug: string;
  model: string;
  org: string;
  provider: string;
  method: "lt" | "b3it";
  date: string;
  daysAgo: number;
  magnitude: number;
  baseline: number;
  start: string;
  end: string;
  values: number[];
  changeFrac: number;
  yMax: number;
}

export interface OverviewData {
  stats: Stats;
  hero: Hero | null;
  feed: FeedItem[];
  providers: ProviderRate[];
  endpoints: EndpointRow[];
}

/** Fetch overview.json; a failure renders the load-error card into `mountId`
 *  before rethrowing, so the page never sits blank. */
export async function loadOverview(mountId: string): Promise<OverviewData> {
  try {
    const res = await fetch("data/overview.json");
    if (!res.ok) throw new Error(`overview.json: HTTP ${res.status}`);
    return await res.json();
  } catch (err) {
    showLoadError(mountId, "the overview data");
    throw err;
  }
}

export const fmtInt = (n: number): string => n.toLocaleString("en-US");
