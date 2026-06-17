from trackllm_website.bi.popularity import aggregate_rankings, map_to_model_ids


def test_aggregate_and_map():
    rows = [
        {"date": "2026-06-16", "model_permaslug": "deepseek/deepseek-v4-flash-20260423", "total_tokens": "100"},
        {"date": "2026-06-15", "model_permaslug": "deepseek/deepseek-v4-flash-20260423", "total_tokens": "50"},
        {"date": "2026-06-16", "model_permaslug": "tencent/hy3-preview-20260421", "total_tokens": "120"},
        {"date": "2026-06-16", "model_permaslug": "other", "total_tokens": "999"},
    ]
    ranked = aggregate_rankings(rows)  # [(permaslug, tokens)] desc, 'other' dropped
    assert ranked[0][0] == "deepseek/deepseek-v4-flash-20260423"  # 150 > 120
    assert all(p != "other" for p, _ in ranked)

    canonical = {"deepseek/deepseek-v4-flash": "deepseek/deepseek-v4-flash",
                 "tencent/hy3-preview": "tencent/hy3-preview"}
    ids = map_to_model_ids([p for p, _ in ranked], canonical)
    assert ids == ["deepseek/deepseek-v4-flash", "tencent/hy3-preview"]


def test_map_strips_variant_suffix_and_dedupes():
    canonical = {"deepseek/deepseek-v4-flash": "deepseek/deepseek-v4-flash"}
    permaslugs = [
        "deepseek/deepseek-v4-flash-20260423",
        "deepseek/deepseek-v4-flash-20260423:free",  # same model, variant -> dedup
    ]
    assert map_to_model_ids(permaslugs, canonical) == ["deepseek/deepseek-v4-flash"]
