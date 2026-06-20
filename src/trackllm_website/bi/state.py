"""Per-endpoint BI monitoring state: epochs of (border inputs, reference samples).

State files record only facts (samples, references, actions taken). Anything
the detection algorithm computes is derived elsewhere and never persisted.
"""

from datetime import datetime
from pathlib import Path
from typing import Literal

import orjson
from pydantic import BaseModel

from trackllm_website.config import Endpoint
from trackllm_website.util import slugify

ReferenceSamples = dict[str, list[tuple[str, str]]]  # prompt -> [(timestamp, token)]


class Epoch(BaseModel):
    start: datetime
    border_inputs: list[str]
    reference: ReferenceSamples
    end: datetime | None = None
    end_reason: Literal["change_detected", "stalled", "gap"] | None = None
    change_date: datetime | None = None
    params: dict | None = None  # detection params in force when the epoch closed


class RetiredInfo(BaseModel):
    reason: Literal["stalled", "no_bis", "delisted"]
    since: datetime
    last_recheck: datetime


class EndpointBIState(BaseModel):
    endpoint: Endpoint
    status: Literal["monitoring", "retired"]
    retired: RetiredInfo | None = None
    epochs: list[Epoch]
    # set when the endpoint drops out of the selected set; cleared if it returns.
    deselected_since: datetime | None = None

    @property
    def slug(self) -> str:
        return slugify(f"{self.endpoint.model}#{self.endpoint.provider}")

    @property
    def current_epoch(self) -> Epoch | None:
        if self.epochs and self.epochs[-1].end is None:
            return self.epochs[-1]
        return None

    def save(self, state_dir: Path) -> None:
        state_dir.mkdir(parents=True, exist_ok=True)
        path = state_dir / f"{self.slug}.json"
        path.write_bytes(
            orjson.dumps(self.model_dump(mode="json"), option=orjson.OPT_INDENT_2)
        )

    @classmethod
    def load(cls, path: Path) -> "EndpointBIState":
        return cls.model_validate(orjson.loads(path.read_bytes()))


def load_all_states(state_dir: Path) -> dict[str, EndpointBIState]:
    if not state_dir.exists():
        return {}
    return {p.stem: EndpointBIState.load(p) for p in sorted(state_dir.glob("*.json"))}
