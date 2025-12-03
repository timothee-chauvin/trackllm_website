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

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
# logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger("trackllm-website")

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


class Config(BaseSettings):
    endpoints_yaml_path: Path = root / "endpoints.yaml"
    model_config = SettingsConfigDict(
        yaml_file=endpoints_yaml_path,
        toml_file=root / "config.toml",
        env_file=root / ".env",
    )

    # read from config.toml
    api: ApiConfig
    prompts: list[str]
    data_dir: Path

    # read from endpoints.yaml
    endpoints: list[Endpoint]

    # read from .env
    openrouter_api_key: str

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
