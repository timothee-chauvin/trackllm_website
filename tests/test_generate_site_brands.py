import pytest

from trackllm_website.generate_site.brands import (
    BRANDS_FILE,
    LOGO_DIR,
    brand_json,
    load_brands,
)


def _write(tmp_path, text: str) -> None:
    (tmp_path / BRANDS_FILE).write_text(text)


def test_missing_file_means_no_brands(tmp_path):
    assert load_brands(tmp_path) == {}
    assert brand_json({}, "acme") == {
        "name": "acme",
        "logo": None,
        "kind": "icon",
        "mono": False,
        "dark": None,
    }


def test_entries_are_validated_and_sources_dropped(tmp_path):
    (tmp_path / LOGO_DIR).mkdir(parents=True)
    (tmp_path / LOGO_DIR / "acme.svg").write_text("<svg/>")
    _write(
        tmp_path,
        'acme:\n  name: "ACME Inc"\n  logo: "acme.svg"\n  kind: "icon"\n'
        '  mono: true\n  dark: null\n  source: "https://acme.example"\n'
        'plain:\n  name: "Plain"\n  logo: null\n',
    )
    brands = load_brands(tmp_path)
    assert brand_json(brands, "acme") == {
        "name": "ACME Inc",
        "logo": "acme.svg",
        "kind": "icon",
        "mono": True,
        "dark": None,
    }
    assert brand_json(brands, "plain")["logo"] is None


def test_a_listed_logo_must_exist(tmp_path):
    _write(tmp_path, 'acme:\n  name: "ACME"\n  logo: "acme.svg"\n')
    with pytest.raises(FileNotFoundError, match="acme.svg"):
        load_brands(tmp_path)


def test_unknown_kind_is_rejected(tmp_path):
    _write(tmp_path, 'acme:\n  name: "ACME"\n  kind: "banner"\n')
    with pytest.raises(ValueError):
        load_brands(tmp_path)
