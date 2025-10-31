import os
from pathlib import Path

from pydantic import BaseModel
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

root = Path(__file__).parent.parent.parent

assert "src" in os.listdir(root)


class Endpoint(BaseModel):
    api: str
    model: str
    provider: str | None = None
    cost: tuple[int | float, int | float]


class Config(BaseSettings):
    model_config = SettingsConfigDict(yaml_file=root / "endpoints.yaml")

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
        return (YamlConfigSettingsSource(settings_cls),)
