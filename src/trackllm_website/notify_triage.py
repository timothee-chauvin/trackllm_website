"""Classify a failed GitHub Actions run as an infra flake or a code failure.

Reads the run's jobs JSON (GET /repos/{repo}/actions/runs/{id}/attempts/{n}/jobs)
on stdin and prints "verdict=infra" or "verdict=code" for $GITHUB_OUTPUT.

Only called for runs that concluded "failure". In an infra failure no step ever
reached conclusion "failure": e.g. "job was not acquired by a runner" leaves the
starved job cancelled with zero steps. Any actually-failed step means our code
failed. A misclassified "infra" costs one retry at most: the attempt-2 failure
always emails.

Pure standard library on purpose, same as notify.py: runs on the system Python
of the notify workflow, before any `uv sync`.
"""

import json
import sys


def classify(jobs: list[dict]) -> str:
    if any(s["conclusion"] == "failure" for j in jobs for s in j["steps"]):
        return "code"
    return "infra"


def main() -> None:
    jobs = json.load(sys.stdin)["jobs"]
    print(f"verdict={classify(jobs)}")


if __name__ == "__main__":
    main()
