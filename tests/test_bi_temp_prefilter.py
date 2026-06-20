from trackllm_website.config import Endpoint
from trackllm_website.update_endpoints import partition_temperature


def ep(model, supports):
    return Endpoint(
        api="openrouter",
        model=model,
        provider="p",
        cost=(1, 1),
        supports_temperature=supports,
    )


def test_partition_skips_temp_unsupported():
    eps = [ep("a", True), ep("b", False), ep("c", None)]
    probe, skip = partition_temperature(eps)
    assert [e.model for e in probe] == ["a", "c"]  # None => probe (unknown)
    assert [e.model for e in skip] == ["b"]  # explicit False => skip
