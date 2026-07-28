"""The contiguous month axis shared by every monthly histogram on the site."""


def month_range(first: str, last: str) -> list[str]:
    """Every "YYYY-MM" from `first`'s month to `last`'s month, inclusive."""
    out, y, m = [], int(first[:4]), int(first[5:7])
    while f"{y:04d}-{m:02d}" <= last[:7]:
        out.append(f"{y:04d}-{m:02d}")
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out
