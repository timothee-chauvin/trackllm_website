import io
import json

from trackllm_website.notify_triage import classify, main


def job(conclusion, steps):
    return {"conclusion": conclusion, "steps": steps}


def step(conclusion):
    return {"conclusion": conclusion}


def test_failed_step_is_code_failure():
    jobs = [job("failure", [step("success"), step("failure")])]
    assert classify(jobs) == "code"


def test_runner_never_acquired_is_infra_failure():
    # Real shape of a "job was not acquired by a runner" failure (run
    # 28993684051): the run concludes failure, but the starved job ends up
    # cancelled with zero steps and the dependent job is skipped.
    jobs = [job("cancelled", []), job("skipped", [])]
    assert classify(jobs) == "infra"


def test_failed_job_without_steps_is_infra_failure():
    jobs = [job("failure", [])]
    assert classify(jobs) == "infra"


def test_failed_job_with_no_failed_step_is_infra_failure():
    jobs = [job("failure", [step("success"), step("skipped")])]
    assert classify(jobs) == "infra"


def test_any_failed_step_wins_over_infra():
    jobs = [
        job("cancelled", []),
        job("failure", [step("failure")]),
    ]
    assert classify(jobs) == "code"


def test_main_reads_jobs_json_and_prints_github_output(monkeypatch, capsys):
    payload = {"jobs": [{"conclusion": "cancelled", "steps": []}]}
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(payload)))
    main()
    assert capsys.readouterr().out == "verdict=infra\n"
