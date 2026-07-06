"""save/load round-trip of phase-2 monthly files.

Regression: JSON round-trips tuples into lists, so appending a second batch
to an existing monthly file made save_results raise a beartype violation
(2026-07-04 monitor run lost a full day of samples for every endpoint).
"""

from trackllm_website.bi.phase_2 import load_existing_results, save_results

REF_BATCH = "2026-07-04T04:39:15+00:00"
NEW_BATCH = "2026-07-05T21:01:00+00:00"


def test_load_existing_results_returns_tuple_samples(tmp_path):
    path = tmp_path / "2026-07.json"
    save_results(path, {"p": {REF_BATCH: [(REF_BATCH, "tok")]}})
    loaded = load_existing_results(path)
    assert loaded["p"][REF_BATCH] == [(REF_BATCH, "tok")]
    assert all(isinstance(s, tuple) for s in loaded["p"][REF_BATCH])


def test_append_second_batch_to_existing_file(tmp_path):
    path = tmp_path / "2026-07.json"
    save_results(path, {"p": {REF_BATCH: [(REF_BATCH, "a"), (REF_BATCH, "b")]}})

    existing = load_existing_results(path)
    existing.setdefault("p", {})[NEW_BATCH] = [(NEW_BATCH, "c")]
    save_results(path, existing)

    loaded = load_existing_results(path)
    assert set(loaded["p"]) == {REF_BATCH, NEW_BATCH}
