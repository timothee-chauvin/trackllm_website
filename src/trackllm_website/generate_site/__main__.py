from pathlib import Path

from trackllm_website.generate_site.render import render_site


def main() -> None:
    render_site(Path("website"))


if __name__ == "__main__":
    main()
