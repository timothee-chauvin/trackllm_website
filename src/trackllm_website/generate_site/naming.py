"""Splitting an OpenRouter provider string into its company and serving variant.

Its own module because both feed.py and provider.py need it, and provider.py
already imports feed.py.
"""


def base_provider(provider: str) -> str:
    """The company -- the part before the "/"."""
    return provider.split("/")[0]


def variant_name(provider: str) -> str:
    """The serving variant, or "" for a provider's default serving stack."""
    return provider.split("/", 1)[1] if "/" in provider else ""
