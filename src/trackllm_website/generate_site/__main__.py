from pathlib import Path

from trackllm_website.config import config
from trackllm_website.generate_site.render import render_site
from trackllm_website.generate_site.status_io import load_status_inputs


def main() -> None:
    render_site(Path("website"), config.hero, load_status_inputs())


if __name__ == "__main__":
    main()
