# TrackLLM Website

Static website for tracking LLM API logprob responses.

## Dependencies

- [uv](https://docs.astral.sh/uv/) - Python package manager
- [Bun](https://bun.sh/) - TypeScript bundler

## Development

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

## Structure

```
website/
├── index.html          # Generated - main page
├── style.css           # Styles
├── js/                 # Generated - compiled TypeScript
├── endpoints/          # Generated - endpoint pages
├── data/               # Query data (written by main app)
├── src/                # TypeScript source
└── templates/          # Jinja2 templates
```

