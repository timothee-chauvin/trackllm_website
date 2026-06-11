from pathlib import Path

import orjson

from trackllm_website.bi.migrate_state import migrate_endpoint
from trackllm_website.config import Endpoint

FIXTURES = Path("tests/fixtures/phase_2")


def test_migrate_builds_closed_gap_epoch():
    slug = "mistralai2fmistral-7b-instruct-v0.323together"
    results = orjson.loads((FIXTURES / slug / "data.json").read_bytes())
    endpoint = Endpoint(
        api="openrouter",
        model="mistralai/mistral-7b-instruct-v0.3",
        provider="together",
        cost=(0.2, 0.2),
    )
    state = migrate_endpoint(endpoint, results)
    assert state.status == "retired"  # all history ends pre-resumption
    [epoch] = state.epochs
    assert epoch.end_reason == "gap"
    assert epoch.start.date().isoformat() == "2026-01-14"
    # end = last day with any successful sample
    assert epoch.end.date().isoformat() == "2026-02-25"
    assert set(epoch.reference) == set(epoch.border_inputs)
