.PHONY: install
# Install dependencies
install:
	cd website && bun install

.PHONY: build
# Build TypeScript and generate HTML
build: install
	cd website && bun run build
	uv run python -m trackllm_website.generate_site

.PHONY: serve
# Serve the website locally
serve: build
	uv run python scripts/serve.py 8000

.PHONY: watch
# Watch TypeScript for changes (run in separate terminal)
watch: install
	cd website && bun run watch

.PHONY: clean
# Clean generated files
clean:
	rm -rf website/js website/endpoints website/models website/orgs website/providers website/feeds website/*.html website/*.md website/llms.txt website/node_modules website/data/overview.json website/data/models website/data/providers

.PHONY: test
# Full suite. The JS smoke tests render the generated site, so they need a build.
test: test-py test-js

.PHONY: test-py
test-py:
	uv run pytest

.PHONY: test-js
test-js: build
	cd website && bun test
