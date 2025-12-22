.PHONY: build serve watch clean install

# Install dependencies
install:
	cd website && bun install

# Build TypeScript and generate HTML
build: install
	cd website && bun run build
	uv run python src/trackllm_website/generate_site.py

# Serve the website locally
serve: build
	cd website && python -m http.server 8000

# Watch TypeScript for changes (run in separate terminal)
watch: install
	cd website && bun run watch

# Clean generated files
clean:
	rm -rf website/js website/endpoints website/index.html website/node_modules
