"""Provider display names and logos: website/provider_brands.yaml.

A provider's OpenRouter slug ("atlas-cloud") is its identity everywhere; the
brand is how it is *shown* -- "Atlas Cloud" beside its logo. Missing entries
fall back to the slug, with no logo, so a new provider never breaks the build.
"""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel

BRANDS_FILE = "provider_brands.yaml"
LOGO_DIR = "logos/providers"


class Brand(BaseModel):
    name: str
    logo: str | None = None
    kind: Literal["icon", "wordmark"] = "icon"
    mono: bool = False
    dark: str | None = None


def load_brands(website_dir: Path) -> dict[str, Brand]:
    path = website_dir / BRANDS_FILE
    if not path.exists():
        return {}
    raw = yaml.safe_load(path.read_text()) or {}
    brands = {
        slug: Brand(**{k: v for k, v in b.items() if k != "source"})
        for slug, b in raw.items()
    }
    for slug, brand in brands.items():
        for f in (brand.logo, brand.dark):
            if f and not (website_dir / LOGO_DIR / f).exists():
                raise FileNotFoundError(
                    f"{BRANDS_FILE}: {slug}: missing {LOGO_DIR}/{f}"
                )
    return brands


def brand_json(brands: dict[str, Brand], slug: str) -> dict:
    return (brands[slug] if slug in brands else Brand(name=slug)).model_dump()
