.PHONY: install
# Install dependencies
install:
	cd website && bun install

.PHONY: build
# Build TypeScript and generate HTML
build: install
	cd website && bun run build
	uv run python src/trackllm_website/generate_site.py

.PHONY: serve
# Serve the website locally
serve: build
	cd website && python -m http.server 8000

.PHONY: watch
# Watch TypeScript for changes (run in separate terminal)
watch: install
	cd website && bun run watch

.PHONY: clean
# Clean generated files
clean:
	rm -rf website/js website/endpoints website/index.html website/node_modules

.PHONY: test
test:
	uv run pytest
