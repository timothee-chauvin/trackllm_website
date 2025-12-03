from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import orjson
from pydantic import BaseModel

from trackllm_website.config import Endpoint
from trackllm_website.util import slugify


class ResponseLogprobs(BaseModel, arbitrary_types_allowed=True):
    """A vector of returned logprobs and the corresponding tokens."""

    tokens: list[str]
    logprobs: list[np.float32]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ResponseLogprobs):
            return False
        return self.tokens == other.tokens and list(self.logprobs) == list(
            other.logprobs
        )


class IdxLogprobVector(BaseModel, arbitrary_types_allowed=True):
    """Like `ResponseLogprobs`, but the tokens are replaced by indices."""

    tokens: list[int]
    logprobs: list[np.float32]


class ResponseError(BaseModel):
    http_code: int
    message: str

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ResponseError):
            return False
        return self.http_code == other.http_code and self.message == other.message


class Response(BaseModel):
    """A response from an LLM API."""

    date: datetime
    endpoint: Endpoint
    prompt: str
    logprobs: ResponseLogprobs | None
    error: ResponseError | None = None
    cost: float | int


class PromptInfo(BaseModel):
    """Metadata stored at the prompt directory level in info.json."""

    prompt: str
    endpoint: Endpoint


class MonthlyData:
    """Results of queries to an LLM API, for a given prompt, in a given month."""

    logprob_filename: str = "logprobs.json"
    queries_filename: str = "queries.json"
    errors_filename: str = "errors.json"

    def __init__(
        self,
        year: int,
        month: int,
        logprob_responses: list[tuple[datetime, ResponseLogprobs]],
        error_responses: list[tuple[datetime, ResponseError]],
    ):
        self.year = year
        self.month = month
        self.logprob_responses = logprob_responses
        self.error_responses = error_responses

    @classmethod
    def load_existing(cls, path: Path, year: int, month: int) -> "MonthlyData":
        """Load existing data from disk if it exists, otherwise return empty MonthlyData."""
        logprobs_path = path / cls.logprob_filename
        queries_path = path / cls.queries_filename
        errors_path = path / cls.errors_filename

        if not logprobs_path.exists() or not queries_path.exists():
            return cls(year=year, month=month, logprob_responses=[], error_responses=[])

        # Load the condensed data
        with open(logprobs_path, "rb") as f:
            logprob_data = orjson.loads(f.read())
        with open(queries_path, "rb") as f:
            queries_data = orjson.loads(f.read())

        # Load errors if they exist
        errors_data = {"seen_errors": []}
        if errors_path.exists():
            with open(errors_path, "rb") as f:
                errors_data = orjson.loads(f.read())

        # Reconstruct ResponseLogprobs objects
        seen_tokens = logprob_data["seen_tokens"]
        seen_logprobs = logprob_data["seen_logprobs"]

        logprob_vectors = []
        for idx_logprob in seen_logprobs:
            tokens = [seen_tokens[i] for i in idx_logprob["tokens"]]
            logprobs = [np.float32(lp) for lp in idx_logprob["logprobs"]]
            logprob_vectors.append(ResponseLogprobs(tokens=tokens, logprobs=logprobs))

        # Reconstruct error objects
        seen_errors = [
            ResponseError(http_code=e["http_code"], message=e["message"])
            for e in errors_data["seen_errors"]
        ]

        # Reconstruct responses
        logprob_responses: list[tuple[datetime, ResponseLogprobs]] = []
        error_responses: list[tuple[datetime, ResponseError]] = []

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
            if isinstance(idx, int):
                # Index of a ResponseLogprob
                logprob_responses.append((full_date, logprob_vectors[idx]))
            else:
                # "e" followed by the index of an error
                err_idx = int(idx[1:])
                error_responses.append((full_date, seen_errors[err_idx]))

        return cls(
            year=year,
            month=month,
            logprob_responses=logprob_responses,
            error_responses=error_responses,
        )

    def merge_response(self, response: Response) -> None:
        """Merge a new response, avoiding duplicates."""
        if response.logprobs is not None:
            # Check for duplicates
            for existing_date, existing_logprobs in self.logprob_responses:
                if (
                    existing_date == response.date
                    and existing_logprobs == response.logprobs
                ):
                    return  # Duplicate, skip
            self.logprob_responses.append((response.date, response.logprobs))
            self.logprob_responses.sort(key=lambda x: x[0])
        elif response.error is not None:
            # Check for duplicates
            for existing_date, existing_error in self.error_responses:
                if existing_date == response.date and existing_error == response.error:
                    return  # Duplicate, skip
            self.error_responses.append((response.date, response.error))
            self.error_responses.sort(key=lambda x: x[0])

    @staticmethod
    def _condense_logprobs(
        logprob_responses: list[tuple[datetime, ResponseLogprobs]],
    ) -> tuple[list[str], list[IdxLogprobVector], list[tuple[str, int]]]:
        """Condense logprob responses into indexed format.

        Returns:
            - seen_tokens: list of unique tokens
            - idx_logprobs: list of IdxLogprobVector with token indices
            - queries: list of (date_str, logprob_idx) tuples
        """
        seen_tokens: list[str] = []
        unique_logprobs: list[ResponseLogprobs] = []
        idx_logprobs: list[IdxLogprobVector] = []
        queries: list[tuple[str, int]] = []

        for date, logprobs in logprob_responses:
            # Find or add the logprob vector
            try:
                logprob_idx = unique_logprobs.index(logprobs)
            except ValueError:
                unique_logprobs.append(logprobs)
                logprob_idx = len(unique_logprobs) - 1

                # Convert tokens to indices
                token_indices = []
                for token in logprobs.tokens:
                    try:
                        token_idx = seen_tokens.index(token)
                    except ValueError:
                        seen_tokens.append(token)
                        token_idx = len(seen_tokens) - 1
                    token_indices.append(token_idx)
                idx_logprobs.append(
                    IdxLogprobVector(tokens=token_indices, logprobs=logprobs.logprobs)
                )

            queries.append((date.strftime("%d %H:%M:%S"), logprob_idx))

        return seen_tokens, idx_logprobs, queries

    @staticmethod
    def _condense_errors(
        error_responses: list[tuple[datetime, ResponseError]],
    ) -> tuple[list[ResponseError], list[tuple[str, str]]]:
        """Condense error responses into indexed format.

        Returns:
            - seen_errors: list of unique errors
            - queries: list of (date_str, "eN") tuples where N is error index
        """
        seen_errors: list[ResponseError] = []
        queries: list[tuple[str, str]] = []

        for date, error in error_responses:
            try:
                error_idx = seen_errors.index(error)
            except ValueError:
                seen_errors.append(error)
                error_idx = len(seen_errors) - 1
            queries.append((date.strftime("%d %H:%M:%S"), f"e{error_idx}"))

        return seen_errors, queries

    def serialize(self, path: Path) -> None:
        """Serialize into JSON files: logprobs.json, queries.json, and errors.json"""
        path.mkdir(parents=True, exist_ok=True)

        # Condense data
        seen_tokens, idx_logprobs, logprob_queries = self._condense_logprobs(
            self.logprob_responses
        )
        seen_errors, error_queries = self._condense_errors(self.error_responses)

        # Merge queries and sort by date
        all_queries: list[tuple[str, int | str]] = logprob_queries + error_queries
        all_queries.sort(key=lambda x: x[0])

        # Write logprobs.json
        with open(path / self.logprob_filename, "wb") as f:
            json_dict = {
                "seen_tokens": seen_tokens,
                "seen_logprobs": [lp.model_dump(mode="python") for lp in idx_logprobs],
            }
            f.write(orjson.dumps(json_dict, option=orjson.OPT_SERIALIZE_NUMPY))

        # Write queries.json
        with open(path / self.queries_filename, "wb") as f:
            f.write(orjson.dumps(all_queries))

        # Write errors.json
        with open(path / self.errors_filename, "wb") as f:
            json_dict = {
                "seen_errors": [e.model_dump() for e in seen_errors],
            }
            f.write(orjson.dumps(json_dict))


class ResultsStorage:
    """Handles storage of API responses to disk in a structured format.

    Directory structure:
        data_dir/
            model_slug/
                prompt_slug/
                    info.json          # Contains endpoint and prompt info
                    YYYY-MM/
                        logprobs.json  # Condensed logprob vectors
                        queries.json   # List of (date, idx) pointing to logprobs or errors
                        errors.json    # List of errors
    """

    info_filename: str = "info.json"

    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def _get_model_slug(self, endpoint: Endpoint) -> str:
        return slugify(f"{endpoint.model}#{endpoint.provider}")

    def _get_prompt_slug(self, prompt: str) -> str:
        return slugify(prompt, max_length=50, hash_length=8)

    def _get_prompt_dir(self, endpoint: Endpoint, prompt: str) -> Path:
        return (
            self.data_dir
            / self._get_model_slug(endpoint)
            / self._get_prompt_slug(prompt)
        )

    def _load_or_create_info(
        self, prompt_dir: Path, endpoint: Endpoint, prompt: str
    ) -> PromptInfo:
        """Load existing info.json or create new PromptInfo."""
        info_path = prompt_dir / self.info_filename
        if info_path.exists():
            with open(info_path, "rb") as f:
                data = orjson.loads(f.read())
            return PromptInfo(**data)
        return PromptInfo(prompt=prompt, endpoint=endpoint)

    def _save_info(self, prompt_dir: Path, info: PromptInfo) -> None:
        """Save info.json to disk."""
        prompt_dir.mkdir(parents=True, exist_ok=True)
        info_path = prompt_dir / self.info_filename
        with open(info_path, "wb") as f:
            f.write(
                orjson.dumps(info.model_dump(mode="python"), option=orjson.OPT_INDENT_2)
            )

    def store_response(self, response: Response) -> None:
        """Store a single API response to storage."""
        # Get directory paths
        prompt_dir = self._get_prompt_dir(response.endpoint, response.prompt)
        year = response.date.year
        month = response.date.month
        month_dir = prompt_dir / f"{year}-{month:02d}"

        # Ensure info.json exists
        info = self._load_or_create_info(prompt_dir, response.endpoint, response.prompt)
        self._save_info(prompt_dir, info)

        # Load existing monthly data
        monthly_data = MonthlyData.load_existing(path=month_dir, year=year, month=month)

        # Merge the new response
        monthly_data.merge_response(response)

        # Serialize back to disk
        monthly_data.serialize(path=month_dir)

    def get_summary(self, responses: list[Response]) -> dict:
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
