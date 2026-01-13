import os
import yaml
from pathlib import Path
from typing import Dict
from dataclasses import dataclass
import logging

logger = logging.getLogger("research_agent")


@dataclass
class DatabaseConfig:
    connection_string: str
    pool_min_size: int = 5
    pool_max_size: int = 20


@dataclass
class LLMConfig:
    base_url: str
    api_key: str
    model_default: str = "gemini-2.5-flash"
    model_pro: str = "gemini-2.5-pro"
    max_retries: int = 3
    timeout: int = 120


@dataclass
class EmbeddingConfig:
    provider: str = "google"
    model: str = "models/text-embedding-004"
    api_key_env: str = "GOOGLE_API_KEY"
    dimensions: int = 768


@dataclass
class ResearchConfig:
    sources_config: str = "configs/research_sources.yaml"
    max_concurrent_fetches: int = 10
    rate_limit: float = 2.0
    max_content_length: int = 1000000


@dataclass
class IngestionConfig:
    max_concurrent_llm_calls: int = 3
    batch_size: int = 10
    retry_backoff_factor: float = 2.0
    max_retries: int = 3


@dataclass
class ProcessingConfig:
    min_entities_to_process: int = 100
    min_time_between_processing_hours: int = 1
    processing_interval_seconds: int = 60


@dataclass
class MonitoringConfig:
    metrics_port: int = 8000
    health_check_interval_seconds: int = 300
    log_retention_days: int = 30


@dataclass
class PathsConfig:
    knowledge_base: str = "\\\\wsl.localhost\\Ubuntu\\home\\administrator\\dev\\unbiased-observer\\knowledge_base"
    cache_dir: str = "./cache"
    logs_dir: str = "./logs"
    state_dir: str = "./state"


@dataclass
class Config:
    database: DatabaseConfig
    llm: LLMConfig
    embedding: EmbeddingConfig
    research: ResearchConfig
    ingestion: IngestionConfig
    processing: ProcessingConfig
    monitoring: MonitoringConfig
    paths: PathsConfig

    def __init__(self, config_path: str = "configs/research_agent_config.yaml"):
        self.config_path = config_path
        self._load_config()

    def _load_config(self):
        config_file = Path(self.config_path)
        if not config_file.exists():
            logger.warning(f"Config file not found: {self.config_path}, using defaults")
            self._set_defaults()
            return

        with open(config_file, "r") as f:
            config_data = yaml.safe_load(f)

        self.database = DatabaseConfig(**config_data.get("database", {}))
        self.llm = LLMConfig(**config_data.get("llm", {}))
        self.embedding = EmbeddingConfig(**config_data.get("embedding", {}))
        self.research = ResearchConfig(**config_data.get("research", {}))
        self.ingestion = IngestionConfig(**config_data.get("ingestion", {}))
        self.processing = ProcessingConfig(**config_data.get("processing", {}))
        self.monitoring = MonitoringConfig(**config_data.get("monitoring", {}))
        self.paths = PathsConfig(**config_data.get("paths", {}))
        self._apply_env_overrides()
        logger.info(f"Configuration loaded from: {self.config_path}")

    def _set_defaults(self):
        self.database = DatabaseConfig(
            connection_string="postgresql://agentzero@localhost:5432/knowledge_graph"
        )
        self.llm = LLMConfig(
            base_url="http://localhost:8317/v1",
            api_key="lm-studio",
            model_default="gemini-2.5-flash",
            model_pro="gemini-2.5-pro",
        )
        self.embedding = EmbeddingConfig(
            provider="google",
            model="models/text-embedding-004",
            api_key_env="GOOGLE_API_KEY",
            dimensions=768,
        )
        self.research = ResearchConfig(
            sources_config="configs/research_sources.yaml",
            max_concurrent_fetches=10,
            rate_limit=2.0,
            max_content_length=1000000,
        )
        self.ingestion = IngestionConfig(
            max_concurrent_llm_calls=3,
            batch_size=10,
            retry_backoff_factor=2.0,
            max_retries=3,
        )
        self.processing = ProcessingConfig(
            min_entities_to_process=100,
            min_time_between_processing_hours=1,
            processing_interval_seconds=60,
        )
        self.monitoring = MonitoringConfig(
            metrics_port=8000, health_check_interval_seconds=300, log_retention_days=30
        )
        self.paths = PathsConfig(
            knowledge_base="\\\\wsl.localhost\\Ubuntu\\home\\administrator\\dev\\unbiased-observer\\knowledge_base",
            cache_dir="./cache",
            logs_dir="./logs",
            state_dir="./state",
        )

    def _apply_env_overrides(self):
        if "DB_CONNECTION_STRING" in os.environ:
            self.database.connection_string = os.environ["DB_CONNECTION_STRING"]
            logger.info("Database connection string overridden from environment")

        if "LLM_BASE_URL" in os.environ:
            self.llm.base_url = os.environ["LLM_BASE_URL"]
            logger.info("LLM base URL overridden from environment")

        if "LLM_API_KEY" in os.environ:
            self.llm.api_key = os.environ["LLM_API_KEY"]
            logger.info("LLM API key overridden from environment")

        if "GOOGLE_API_KEY" in os.environ:
            self.embedding.api_key_env = "GOOGLE_API_KEY"
            logger.info("Google API key environment variable set")

    def get_env_value(self, env_var: str, default: str = None) -> str:
        return os.getenv(env_var, default)


config = None


def load_config(config_path: str = None) -> Config:
    global config
    if config is None:
        config = Config(config_path or "configs/research_agent_config.yaml")
    return config
