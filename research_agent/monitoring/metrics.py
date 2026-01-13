"""
Metrics and Logging Module

Provides structured logging and metrics collection for the research agent.
"""

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..config import Config


def setup_logging(
    config: "Config",
    debug: bool = False,
) -> tuple[logging.Logger, logging.Logger, logging.Logger, logging.Logger]:
    """
    Setup structured logging with both file and console output.

    Args:
        config: Configuration object
        debug: Enable debug logging (overrides config if needed)

    Returns:
        Tuple of (agent_logger, ingestion_logger, processing_logger, orchestrator_logger)
    """
    log_dir = Path(config.paths.logs_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_level = logging.DEBUG if debug else logging.INFO

    logging.basicConfig(
        format=log_format,
        level=log_level,
        force=True,
    )

    agent_logger = logging.getLogger("research_agent")
    ingestion_logger = logging.getLogger("research_agent.ingestion")
    processing_logger = logging.getLogger("research_agent.processing")
    orchestrator_logger = logging.getLogger("research_agent.orchestrator")

    for logger_name, logger in [
        ("agent", agent_logger),
        ("ingestion", ingestion_logger),
        ("processing", processing_logger),
        ("orchestrator", orchestrator_logger),
    ]:
        log_file = log_dir / f"{logger_name}.log"
        handler = logging.FileHandler(log_file)
        handler.setLevel(log_level)
        handler.setFormatter(logging.Formatter(log_format))
        logger.addHandler(handler)
        logger.setLevel(log_level)

    return agent_logger, ingestion_logger, processing_logger, orchestrator_logger


class MetricsCollector:
    """Collects and reports metrics for the research agent."""

    def __init__(self, config: "Config", logger: logging.Logger):
        self.config = config
        self.logger = logger

    async def record_task_start(self, task_id: str, task_type: str) -> None:
        """Record the start of a task."""
        self.logger.info(f"Task started - ID: {task_id}, Type: {task_type}")

    async def record_task_complete(self, task_id: str, duration_seconds: float) -> None:
        """Record the completion of a task."""
        self.logger.info(
            f"Task completed - ID: {task_id}, Duration: {duration_seconds:.2f}s",
        )

    async def record_task_failure(
        self,
        task_id: str,
        error: Exception,
        retry_count: int,
    ) -> None:
        """Record a task failure."""
        self.logger.error(
            f"Task failed - ID: {task_id}, Error: {str(error)}, "
            f"Type: {type(error).__name__}, Retry: {retry_count}",
            exc_info=True,
        )

    async def record_ingestion_start(self, source_url: str) -> None:
        """Record the start of an ingestion."""
        self.logger.info(f"Ingestion started - URL: {source_url}")

    async def record_ingestion_complete(
        self,
        source_url: str,
        entities_count: int,
        relationships_count: int,
        duration_seconds: float,
    ) -> None:
        """Record the completion of an ingestion."""
        self.logger.info(
            f"Ingestion completed - URL: {source_url}, "
            f"Entities: {entities_count}, Relationships: {relationships_count}, "
            f"Duration: {duration_seconds:.2f}s",
        )

    async def record_processing_start(self, job_id: str) -> None:
        """Record the start of processing."""
        self.logger.info(f"Processing started - Job ID: {job_id}")

    async def record_processing_complete(
        self,
        job_id: str,
        duration_seconds: float,
    ) -> None:
        """Record the completion of processing."""
        self.logger.info(
            f"Processing completed - Job ID: {job_id}, Duration: {duration_seconds:.2f}s",
        )

    async def get_summary_metrics(self) -> dict[str, Any]:
        """Get summary metrics for monitoring."""
        return {
            "pending_tasks": 0,
            "failed_tasks": 0,
            "recent_activity": [],
        }
