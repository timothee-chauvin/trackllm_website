import logging
import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, model_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
    YamlConfigSettingsSource,
)

root = Path(__file__).parent.parent.parent

assert "src" in os.listdir(root)


class Endpoint(BaseModel):
    api: Literal["openrouter"]
    model: str
    provider: str | None = None
    cost: tuple[int | float, int | float]
    max_logprobs: int | None = None

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Endpoint):
            return False
        return (self.api, self.model, self.provider) == (
            other.api,
            other.model,
            other.provider,
        )

    def __hash__(self) -> int:
        return hash((self.api, self.model, self.provider))

    def __str__(self) -> str:
        return f"{self.api}#{self.model}#{self.provider}"

    @model_validator(mode="after")
    def check_provider(self) -> "Endpoint":
        if self.api == "openrouter" and self.provider is None:
            raise ValueError("provider is required when api is 'openrouter'")
        return self

    def get_max_logprobs(self, cfg: "Config") -> int:
        # First case: max_logprobs is part of the endpoint
        if self.max_logprobs is not None:
            return self.max_logprobs
        # Second case: the provider is known
        for provider_prefix in cfg.api.top_logprobs_openrouter.keys():
            if self.provider.lower().startswith(provider_prefix.lower()):
                return cfg.api.top_logprobs_openrouter[provider_prefix]
        # Third case: openrouter default
        return cfg.api.top_logprobs_openrouter_default

    @property
    def provider_without_suffix(self) -> str:
        return self.provider.split("/")[0]


class Phase1Config(BaseModel):
    queries_per_token: int
    requests_per_second_per_endpoint: float
    tokens_per_endpoint: int
    max_concurrent_requests_per_endpoint: int
    max_concurrent_tokens_per_endpoint: int
    request_delay_seconds: float
    border_input_candidate_ratio: float
    target_border_inputs: int
    queries_per_candidate: int


class Phase2Config(BaseModel):
    queries_per_token: int
    requests_per_second_per_endpoint: float
    max_concurrent_requests_per_endpoint: int
    request_delay_seconds: float


class BIConfig(BaseModel):
    data_dir: Path
    phase_1: Phase1Config
    phase_2: Phase2Config

    @property
    def tokenizers_dir(self) -> Path:
        return self.data_dir / "tokenizers"

    def get_phase_1_dir(self, temperature: float | int) -> Path:
        if temperature == 0:
            return self.data_dir / "phase_1" / "T=0"
        return self.data_dir / "phase_1" / f"T={temperature:g}"

    @property
    def phase_2_dir(self) -> Path:
        return self.data_dir / "phase_2"


class ApiConfig(BaseModel):
    # Default is 20, but some providers have a lower limit. See Endpoint.get_max_logprobs().
    top_logprobs_openrouter: dict[str, int]
    top_logprobs_openrouter_default: int
    temperature: float
    max_retries: int
    max_cost_mtok: float
    max_workers: int
    timeout: float
    abandon_after: int
    openrouter_avoid_free_endpoints: bool


class Config(BaseSettings):
    endpoints_yaml_path_lt: Path = root / "endpoints_lt.yaml"
    endpoints_yaml_path_bi: Path = root / "endpoints_bi.yaml"
    endpoints_yaml_path_bi_phase_1: Path = root / "endpoints_bi_phase_1.yaml"
    model_config = SettingsConfigDict(
        yaml_file=[
            endpoints_yaml_path_lt,
            endpoints_yaml_path_bi,
            endpoints_yaml_path_bi_phase_1,
        ],
        toml_file=root / "config.toml",
        env_file=root / ".env",
    )

    # read from config.toml
    api: ApiConfig
    bi: BIConfig
    prompts: list[str]
    data_dir: Path
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

    # read from endpoints_....yaml
    endpoints_lt: list[Endpoint]
    endpoints_bi: list[Endpoint]
    endpoints_bi_phase_1: list[Endpoint]

    # read from .env
    openrouter_api_key: str
    hf_token: str | None = None

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            env_settings,  # Read from environment variables first
            dotenv_settings,
            YamlConfigSettingsSource(settings_cls),
            TomlConfigSettingsSource(settings_cls),
        )


config = Config()

logging.basicConfig(
    level=config.log_level,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
# logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger("trackllm-website")
