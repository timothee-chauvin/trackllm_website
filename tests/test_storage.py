from trackllm_website.config import Endpoint
from trackllm_website.storage import ResultsStorage
from trackllm_website.util import slugify

ENDPOINT = Endpoint(api="openrouter", model="org/model", provider="prov", cost=(1, 1))


def test_is_stalled_ignores_non_directory_artifacts(tmp_path):
    """Regression: an `lt_scores.json` file lives in the endpoint dir alongside the
    per-prompt directories. The stall check globs `<slug>/*`, so it must skip
    non-directories instead of iterdir()-ing a file (NotADirectoryError), which
    crashed the daily update-endpoints job for days."""
    storage = ResultsStorage(tmp_path)
    model_dir = tmp_path / slugify(f"{ENDPOINT.model}#{ENDPOINT.provider}")
    model_dir.mkdir(parents=True)
    (model_dir / "lt_scores.json").write_text("{}")  # the artifact that broke it
    (model_dir / "someprompt").mkdir()  # a real prompt directory

    assert storage.is_stalled(ENDPOINT) is False
