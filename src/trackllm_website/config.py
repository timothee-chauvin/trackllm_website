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
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
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

    @model_validator(mode="after")
    def check_provider(self) -> "Endpoint":
        if self.api == "openrouter" and self.provider is None:
            raise ValueError("provider is required when api is 'openrouter'")
        return self

    def get_max_logprobs(self, cfg: "Config") -> int:
        if self.max_logprobs is not None:
            return self.max_logprobs
        for provider_prefix in cfg.api.top_logprobs_openrouter.keys():
            if self.provider.lower().startswith(provider_prefix.lower()):
                return cfg.api.top_logprobs_openrouter[provider_prefix]


class ApiConfig(BaseModel):
    # Default is 20, but some providers have a lower limit. See Endpoint.get_max_logprobs().
    top_logprobs_openrouter: dict[str, int]


class Config(BaseSettings):
    model_config = SettingsConfigDict(
        yaml_file=root / "endpoints.yaml", toml_file=root / "config.toml"
    )

    # read from config.toml
    api: ApiConfig

    # read from endpoints.yaml
    endpoints: list[Endpoint]

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
            YamlConfigSettingsSource(settings_cls),
            TomlConfigSettingsSource(settings_cls),
        )


config = Config()
