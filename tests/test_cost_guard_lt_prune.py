"""LT fleet pruning by measured per-query ledger cost (the durable removal for
endpoints that trip the per-query guard: run-main.yml commits only website/data)."""

import pytest
import orjson
from datetime import datetime, timezone

from trackllm_website.config import Endpoint, config
from trackllm_website.update_endpoints import lt_cost_per_query, prune_expensive_lt

NOW = datetime(2026, 2, 15, tzinfo=timezone.utc)


def ep(model="m/x", provider="p"):
    return Endpoint(api="openrouter", model=model, provider=provider, cost=(1, 1))


@pytest.fixture
def guard(monkeypatch):
    monkeypatch.setattr(config.api, "max_cost_per_query", 1e-4)


def _write_ledger(spend_dir, slug, month, entries):
    d = spend_dir / slug
    d.mkdir(parents=True, exist_ok=True)
    with open(d / f"{month}.jsonl", "ab") as f:
        for kind, cost, nq, ne in entries:
            f.write(
                orjson.dumps(
                    {
                        "timestamp": f"{month}-10T00:00:00Z",
                        "kind": kind,
                        "cost": cost,
                        "n_queries": nq,
                        "n_errors": ne,
                    }
                )
                + b"\n"
            )


def test_lt_cost_per_query_from_ledger(tmp_path):
    _write_ledger(tmp_path, "slug", "2026-02", [("lt", 0.5, 40, 15)])
    _write_ledger(
        tmp_path, "slug", "2026-01", [("lt", 0.5, 25, 0), ("monitor", 9, 200, 0)]
    )
    # 1.0 / (25 + 25 successful lt queries); monitor lines excluded
    assert lt_cost_per_query(tmp_path, "slug", NOW) == pytest.approx(0.02)


def test_lt_cost_per_query_needs_enough_queries(tmp_path):
    _write_ledger(tmp_path, "slug", "2026-02", [("lt", 0.5, 19, 0)])
    assert lt_cost_per_query(tmp_path, "slug", NOW) is None


def test_prune_expensive_lt(tmp_path, guard, monkeypatch):
    from trackllm_website.util import slugify

    pricey, cheap, unknown = ep("m/pricey"), ep("m/cheap"), ep("m/unknown")
    _write_ledger(
        tmp_path,
        slugify(f"{pricey.model}#{pricey.provider}"),
        "2026-02",
        [("lt", 0.1, 50, 0)],
    )
    _write_ledger(
        tmp_path,
        slugify(f"{cheap.model}#{cheap.provider}"),
        "2026-02",
        [("lt", 0.001, 50, 0)],
    )
    kept, failures = prune_expensive_lt([pricey, cheap, unknown], tmp_path, NOW)
    assert kept == [cheap, unknown]
    assert list(failures) == [pricey]
    assert failures[pricey] == "too_expensive: $0.002000/query"
