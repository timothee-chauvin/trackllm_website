# TrackLLM Website

Static website for tracking LLM API logprob responses.

## Dependencies

- [uv](https://docs.astral.sh/uv/) - Python package manager
- [Bun](https://bun.sh/) - TypeScript bundler

## Development

Raw monitoring data lives in [trackllm_data](https://github.com/timothee-chauvin/trackllm_data),
expected at `website/data` (gitignored here). Clone it there once — or symlink an
existing clone, one per worktree:

```bash
git clone git@github.com:timothee-chauvin/trackllm_data website/data
```

```bash
# Build and serve locally
make serve

# Just build (no server)
make build

# Watch TypeScript for changes (run in separate terminal)
make watch

# Clean generated files
make clean
```

The site will be available at http://localhost:8000

## Testing

```bash
make test       # both suites
make test-py    # pytest
make test-js    # bun test (renders the generated site, so it builds first)
```

## Structure

Source:

```
src/trackllm_website/
├── main.py             # LT collection: query the tracked endpoints, store responses
├── bi/                 # border inputs: selection, vetting, phase 1/2, detection, monitor
├── generate_site/      # the static site generator (page JSON + Jinja rendering)
├── update_endpoints.py # refresh the OpenRouter catalog and the vetting caches
├── lt_*.py             # LT scores, drift and changepoints from the stored responses
└── storage.py          # on-disk format for collected responses

tests/                  # pytest suite for everything above
config.toml             # single source for paths, budgets and thresholds
endpoints_*.yaml        # tracked fleets, catalog snapshot and vetting caches (committed data)
```

Site:

```
website/
├── src/                # TypeScript, one entrypoint per page kind
├── templates/          # Jinja2 templates (base + one per page kind)
├── style.css           # the whole design system
├── test/               # bun tests: render the generated pages, check links and a11y
├── data/               # the trackllm_data repo (see Development) + generated page JSON
│   ├── lt/             # collected logprob responses, per endpoint/prompt/month
│   ├── b3it/           # border-input state, phase-2 samples, per-endpoint views
│   ├── spend/          # per-endpoint spend ledgers
│   └── *.json          # generated, untracked: overview, changes, changes_page, models/, providers/
└── (generated pages)   # index, changes, spend, methodology + endpoints/ models/ providers/ orgs/
```

## Data pipeline

The site is a view over data four scheduled GitHub Actions workflows collect and
commit to [trackllm_data](https://github.com/timothee-chauvin/trackllm_data)
(endpoint fleets and vetting caches stay in this repo):

| Workflow | Schedule | What it does |
| --- | --- | --- |
| [run-main](.github/workflows/run-main.yml) | hourly | queries the LT fleet, stores responses, recomputes LT scores |
| [bi-monitor](.github/workflows/bi-monitor.yml) | daily | samples the border inputs of every monitored endpoint, detects changes |
| [update-endpoints](.github/workflows/update-endpoints.yml) | daily | refreshes the OpenRouter catalog, vets and selects endpoints |
| [deploy-pages](.github/workflows/deploy-pages.yml) | after run-main | builds the site and deploys it to GitHub Pages |

[notify-on-failure](.github/workflows/notify-on-failure.yml) watches all four and
emails when one fails.

