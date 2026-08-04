#!/usr/bin/env bash
# Strip everything the deployed site never serves from the Pages artifact dir,
# before upload-pages-artifact tars it. Without this, the artifact ships
# node_modules plus ~500MB of raw pipeline data under website/data, which the
# browser never fetches and which would eventually push the site past GitHub
# Pages' 1GB limit.
#
# The full list of data/ paths the site fetches lives in website/src/*.ts
# (grep for `fetch(`). If a new fetched file is added under data/lt/<slug>/ or
# data/b3it/, update this script, or the sanity checks below will only catch
# the breakage if the file is missing everywhere.
set -euo pipefail

site="$1"

rm -rf "$site/node_modules"

# The data checkout at data/ brings its own repository metadata.
rm -rf "$site/data/.git"

# Pipeline spend ledger; the site only fetches the aggregated data/spend.json.
rm -rf "$site/data/spend"

# data/b3it: raw pipeline state. Keep the per-endpoint <slug>/b3it.json dirs.
rm -rf "$site"/data/b3it/{bi_prevalence,logprob_stats,onboarding_progress,phase_1,phase_2,state,tokenizers,scan_backfill.json}

# data/lt/<slug>/: keep lt_scores.json, drop the raw per-prompt response dirs
# (arbitrary names derived from the prompt, so no stable exclude list exists).
find "$site/data/lt" -mindepth 2 -maxdepth 2 ! -name lt_scores.json -exec rm -rf {} +

# Fail the deploy loudly if pruning ever removes something the site fetches.
test -f "$site/index.html"
test -f "$site/data/overview.json"
test -f "$site/data/spend.json"
test -f "$site/data/changes_page.json"
test -n "$(find "$site/data/models" -name '*.json' -print -quit)"
test -n "$(find "$site/data/providers" -name '*.json' -print -quit)"
test -n "$(find "$site/data/lt" -name lt_scores.json -print -quit)"
test -n "$(find "$site/data/b3it" -name b3it.json -print -quit)"
