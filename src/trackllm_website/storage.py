from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import orjson
from pydantic import BaseModel

from trackllm_website.config import Endpoint
from trackllm_website.util import slugify


class ResponseLogprobs(BaseModel, arbitrary_types_allowed=True):
    """A vector of returned logprobs and the corresponding tokens. May be returned to multiple queries if non-determinism is low."""

    tokens: list[str]
    logprobs: list[np.float32]


class IdxLogprobVector(BaseModel, arbitrary_types_allowed=True):
    """Like `ResponseLogprobs`, but the tokens are replaced by indices."""

    tokens: list[int]
    logprobs: list[np.float32]


class ResponseError(BaseModel):
    http_code: int
    message: str


class Response(BaseModel):
    """A response from an LLM API."""

    date: datetime
    endpoint: Endpoint
    prompt: str
    logprobs: ResponseLogprobs | None
    error: ResponseError | None = None
    cost: float | int


class MonthlyData:
    """Results of queries to an LLM API, for a given prompt, in a given month."""

    logprob_filename: str = "logprobs.json"
    queries_filename: str = "queries.json"
    errors_filename: str = "errors.json"

    def __init__(
        self,
        prompt: str,
        year: int,
        month: int,
        responses: list[ResponseLogprobs],
        errors: list[ResponseError],
    ):
        self.prompt = prompt
        self.year = year
        self.month = month
        self.responses = responses
        self.errors = errors

    @classmethod
    def load_existing(
        cls, path: Path, prompt: str, year: int, month: int
    ) -> "MonthlyData":
        """Load existing data from disk if it exists, otherwise return empty MonthlyData."""
        logprobs_path = path / cls.logprob_filename
        queries_path = path / cls.queries_filename
        errors_path = path / cls.errors_filename

        if (
            not logprobs_path.exists()
            or not queries_path.exists()
            or not errors_path.exists()
        ):
            return cls(prompt=prompt, year=year, month=month, responses=[], errors=[])

        # Load the condensed data
        with open(logprobs_path, "rb") as f:
            logprob_data = orjson.loads(f.read())
        with open(queries_path, "rb") as f:
            queries_data = orjson.loads(f.read())
        with open(errors_path, "rb") as f:
            errors_data = orjson.loads(f.read())

        # Reconstruct Response objects
        seen_tokens = logprob_data["seen_tokens"]
        seen_logprobs = logprob_data["seen_logprobs"]

        seen_errors = errors_data["seen_errors"]

        logprob_vectors = []
        for idx_logprob in seen_logprobs:
            tokens = [seen_tokens[i] for i in idx_logprob["tokens"]]
            logprobs = [np.float32(lp) for lp in idx_logprob["logprobs"]]
            logprob_vectors.append(ResponseLogprobs(tokens=tokens, logprobs=logprobs))

        responses = []
        for date_str, idx in queries_data:
            # Parse date from "dd HH:MM:SS" format using the known year and month
            day_time = datetime.strptime(date_str, "%d %H:%M:%S")
            full_date = datetime(
                year,
                month,
                day_time.day,
                day_time.hour,
                day_time.minute,
                day_time.second,
                tzinfo=timezone.utc,
            )
            logprobs = None
            error = None
            if isinstance(idx, int):
                # Index of a ResponseLogprob
                logprobs = logprob_vectors[idx]
            else:
                # "e" followed by the index of an error
                err_idx = int(idx[1:])
                error = seen_errors[err_idx]
            responses.append(
                Response(
                    date=full_date,
                    endpoint=endpoint,
                    prompt=prompt,
                    logprobs=logprobs,
                    error=error,
                )
            )

        return cls(prompt=prompt, year=year, month=month, responses=responses)

    def merge_and_deduplicate(self, new_responses: list[ResponseLogprobs]) -> None:
        """Merge new responses with existing ones, remove duplicates, and sort by date."""
        # Create a set of existing response signatures to detect duplicates
        existing_sigs = {
            (r.date, tuple(r.logprob_vector.tokens), tuple(r.logprob_vector.logprobs))
            for r in self.responses
        }

        # Add new responses that don't already exist
        for response in new_responses:
            sig = (
                response.date,
                tuple(response.logprob_vector.tokens),
                tuple(response.logprob_vector.logprobs),
            )
            if sig not in existing_sigs:
                self.responses.append(response)
                existing_sigs.add(sig)

        # Sort all responses by date
        self.responses.sort(key=lambda r: r.date)

    @staticmethod
    def _condense_responses(
        responses: list[ResponseLogprobs],
    ) -> tuple[list[LogprobVector], list[tuple[str, int]]]:
        """Return a list of the unique logprob vectors, and a list of queries with shortened dates and using indices to the logprob vectors instead of the full logprob vectors."""
        logprobs = []
        queries = []
        for response in responses:
            try:
                idx = logprobs.index(response.logprob_vector)
            except ValueError:
                logprobs.append(response.logprob_vector)
                idx = len(logprobs) - 1
            queries.append((response.date.strftime("%d %H:%M:%S"), idx))
        return logprobs, queries

    @staticmethod
    def _condense_logprobs(
        logprobs: list[LogprobVector],
    ) -> tuple[list[str], list[IdxLogprobVector]]:
        tokens = []
        idx_logprobs = []
        for logprob in logprobs:
            token_indices = []
            for token in logprob.tokens:
                try:
                    idx = tokens.index(token)
                except ValueError:
                    tokens.append(token)
                    idx = len(tokens) - 1
                token_indices.append(idx)
            idx_logprobs.append(
                IdxLogprobVector(tokens=token_indices, logprobs=logprob.logprobs)
            )
        return tokens, idx_logprobs

    def serialize(self, path: Path):
        """Serialize into two JSON files: `self.logprob_filename` and `self.queries_filename`"""
        path.mkdir(parents=True, exist_ok=True)
        logprobs, queries = self._condense_responses(self.responses)
        tokens, idx_logprobs = self._condense_logprobs(logprobs)
        with open(path / self.logprob_filename, "wb") as f:
            json_dict = {
                "seen_tokens": tokens,
                "seen_logprobs": [lp.model_dump(mode="python") for lp in idx_logprobs],
            }
            f.write(orjson.dumps(json_dict, option=orjson.OPT_SERIALIZE_NUMPY))
        with open(path / self.queries_filename, "wb") as f:
            f.write(orjson.dumps(queries))


class ResultsStorage:
    """Handles storage of API responses to disk in a structured format."""

    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def store_response(self, response: Response):
        """Store a single API response to storage."""
        # Create directory structure: data_dir / model_provider / year / month
        model_slug = slugify(f"{response.endpoint.model}#{response.endpoint.provider}")
        year = response.logprobs.date.year
        month = response.logprobs.date.month

        month_path = self.data_dir / model_slug / str(year) / f"{month:02d}"

        # Load existing data for this month and prompt
        monthly_data = MonthlyData.load_existing(
            path=month_path, prompt=response.prompt, year=year, month=month
        )

        # Merge and deduplicate
        monthly_data.merge_and_deduplicate([response.logprobs])

        # Serialize back to disk
        monthly_data.serialize(path=month_path)

    def get_summary(self, responses: list) -> dict:
        """Generate a summary of responses grouped by endpoint.

        Args:
            responses: List of Response objects

        Returns:
            Dictionary mapping endpoint identifiers to summary statistics
        """
        summary = {}
        for response in responses:
            key = f"{response.endpoint.model}#{response.endpoint.provider}"
            if key not in summary:
                summary[key] = {
                    "success": 0,
                    "error": 0,
                    "total_cost": 0.0,
                }

            if response.error:
                summary[key]["error"] += 1
            else:
                summary[key]["success"] += 1

            summary[key]["total_cost"] += response.cost

        return summary
