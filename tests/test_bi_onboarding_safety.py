from trackllm_website.config import config


def test_onboarding_safety_knobs_present():
    assert config.bi.phase_1.max_retries == 3            # 4 attempts (onboarding)
    assert config.bi.phase_1.abandon_after_timeouts == 20
    assert config.bi.reinit.onboard_timeout_seconds == 10800  # 3h
    assert config.bi.reinit.onboard_concurrency == 40
