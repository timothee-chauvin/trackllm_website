from trackllm_website.generate_site.changes import merge_changes


def test_merge_sorts_newest_first_across_methods():
    lt_changes = {"s1": [{"endpoint": "s1", "index": 5, "date": "2026-03-01T00:00:00Z",
                          "sigma": 12.0, "first_detected": "2026-06-01T00:00:00Z"}]}
    class V:
        model = "m/b"; provider = "q"
        epochs = [{"start": "2026-01-01T00:00:00Z", "end": "2026-05-01T00:00:00Z",
                   "end_reason": "change_detected", "change_date": "2026-04-15T00:00:00Z"}]
    class LT:
        model = "m/a"; provider = "p"
    events = merge_changes(lt_changes, {"s1": LT()}, {"s2": V()})
    assert [e.method for e in events] == ["B3IT", "LT"]  # 2026-04-15 > 2026-03-01
    assert events[0].slug == "s2"
    assert events[1].magnitude == 12.0


def test_empty_inputs_yield_empty_feed():
    assert merge_changes({}, {}, {}) == []
