from datetime import datetime
from pathlib import Path

import numpy as np
import orjson
from pydantic import BaseModel

from trackllm_website.api import LogprobResponse, LogprobVector


class IdxLogprobVector(BaseModel, arbitrary_types_allowed=True):
    """Like `LogprobVector`, but the tokens are replaced by indices."""

    tokens: list[int]
    logprobs: list[np.float32]


class MonthlyData:
    """Results of queries to an LLM API, for a given prompt, in a given month."""

    logprob_filename: str = "logprobs.json"
    queries_filename: str = "queries.json"

    def __init__(self, prompt: str, responses: list[LogprobResponse]):
        self.prompt = prompt
        self.responses = responses

    @staticmethod
    def _condense_responses(
        responses: list[LogprobResponse],
    ) -> tuple[list[LogprobVector], list[tuple[datetime, int]]]:
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
                "tokens": tokens,
                "logprobs": [lp.model_dump(mode="python") for lp in idx_logprobs],
            }
            f.write(orjson.dumps(json_dict, option=orjson.OPT_SERIALIZE_NUMPY))
        with open(path / self.queries_filename, "wb") as f:
            f.write(orjson.dumps(queries))
