from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    registry_root: str = "/registry"

    dvc_registry_root: str = "/app/storage/registry"
    dvc_enabled: bool = True
    dvc_remote_name: str = "origin"
    dvc_auto_push: bool = True

    uploads_dir_name: str = "uploads"
    production_dir_name: str = "production"
    archive_dir_name: str = "archive"

    registry_api_token: str = "change_me"
    comparison_primary_metric: str = "test_f1"
    comparison_fallback_metric: str = "test_accuracy"

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )


settings = Settings()