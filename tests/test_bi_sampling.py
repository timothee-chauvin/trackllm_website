import asyncio
from datetime import datetime, timezone

from trackllm_website.bi.common import PlainStrategy
from trackllm_website.bi.sampling import sample_prompts
from trackllm_website.config import Endpoint
from trackllm_website.storage import Response


# `extract_first_token` is beartyped to require a real `Response`, so we build
# one (content carries the token; reasoning_content stays None).
def make_response(endpoint, prompt, content):
    return Response(
        date=datetime.now(tz=timezone.utc),
        endpoint=endpoint,
        prompt=prompt,
        content=content,
        cost=0.0,
        error=None,
    )


class FakeClient:
    def __init__(self, answers):
        self.answers = answers  # prompt -> iterator of tokens

    async def query(self, endpoint, prompt, **kwargs):
        return make_response(endpoint, prompt, next(self.answers[prompt]))


def test_sample_prompts_collects_n_per_prompt(monkeypatch):
    from trackllm_website.config import config

    monkeypatch.setattr(config.bi.phase_2, "request_delay_seconds", 0.0)
    endpoint = Endpoint(api="openrouter", model="m/x", provider="p", cost=(1, 1))
    client = FakeClient({"a": iter("xyxyx"), "b": iter("zzzzz")})
    samples, n_errors = asyncio.run(
        sample_prompts(
            client, endpoint, PlainStrategy(), ["a", "b"], 3, temperature=0.0
        )
    )
    assert n_errors == 0
    assert [tok for _, tok in samples["a"]] == ["x", "y", "x"]
    assert len(samples["b"]) == 3


class RaisingClient:
    def __init__(self, answers, raising_prompt):
        self.answers = answers
        self.raising_prompt = raising_prompt

    async def query(self, endpoint, prompt, **kwargs):
        if prompt == self.raising_prompt:
            raise RuntimeError("boom")
        return make_response(endpoint, prompt, next(self.answers[prompt]))


def test_sample_prompts_survives_per_prompt_exception(monkeypatch):
    from trackllm_website.config import config

    monkeypatch.setattr(config.bi.phase_2, "request_delay_seconds", 0.0)
    endpoint = Endpoint(api="openrouter", model="m/x", provider="p", cost=(1, 1))
    client = RaisingClient({"a": iter("xyxyx")}, raising_prompt="b")
    samples, n_errors = asyncio.run(
        sample_prompts(
            client, endpoint, PlainStrategy(), ["a", "b"], 3, temperature=0.0
        )
    )
    assert [tok for _, tok in samples["a"]] == ["x", "y", "x"]
    assert n_errors > 0
