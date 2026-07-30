from trackllm_website.update_endpoints import parse_model_endpoints


def raw_endpoint(tag, supported_parameters=...):
    e = {"tag": tag, "pricing": {"prompt": "0.000001", "completion": "0.000002"}}
    if supported_parameters is not ...:
        e["supported_parameters"] = supported_parameters
    return e


def parse(raw, model_supports_temperature):
    return parse_model_endpoints(
        raw,
        model_id="org/m",
        created=None,
        model_supports_temperature=model_supports_temperature,
        logprob_filter=False,
    )


def test_endpoint_level_supported_parameters_win():
    # Model-level flag is the union across endpoints: a mixed model claims
    # temperature even when most endpoints don't support it.
    raw = [
        raw_endpoint("azure", ["temperature", "max_tokens"]),
        raw_endpoint("anthropic", ["max_tokens"]),
    ]
    by_provider = {e.provider: e for e in parse(raw, model_supports_temperature=True)}
    assert by_provider["azure"].supports_temperature is True
    assert by_provider["anthropic"].supports_temperature is False


def test_fallback_to_model_flag_when_endpoint_field_missing():
    raw = [raw_endpoint("p")]
    assert parse(raw, model_supports_temperature=True)[0].supports_temperature is True
    assert parse(raw, model_supports_temperature=False)[0].supports_temperature is False


def test_parse_stamps_cost_and_identity():
    (e,) = parse([raw_endpoint("p", ["temperature"])], model_supports_temperature=False)
    assert (e.api, e.model, e.provider) == ("openrouter", "org/m", "p")
    assert e.cost == (1.0, 2.0)


def test_parse_stamps_supports_logprobs():
    raw = [
        raw_endpoint("both", ["logprobs", "top_logprobs"]),
        raw_endpoint("partial", ["logprobs"]),
        raw_endpoint("none", []),
        raw_endpoint("absent"),
    ]
    by_provider = {e.provider: e for e in parse(raw, model_supports_temperature=True)}
    assert by_provider["both"].supports_logprobs is True
    assert by_provider["partial"].supports_logprobs is False
    assert by_provider["none"].supports_logprobs is False
    assert by_provider["absent"].supports_logprobs is None


def test_logprob_filter_uses_endpoint_parameters():
    raw = [
        raw_endpoint("with", ["logprobs", "top_logprobs", "temperature"]),
        raw_endpoint("without", ["temperature"]),
    ]
    parsed = parse_model_endpoints(
        raw,
        model_id="org/m",
        created=None,
        model_supports_temperature=True,
        logprob_filter=True,
    )
    assert [e.provider for e in parsed] == ["with"]
