import re
from pathlib import Path
from research_agent.research.source_discovery import SourceDiscovery
from research_agent.orchestrator.task_queue import TaskQueue
import logging

logger = logging.getLogger("research_agent.research")


class ManualSourceManager:
    """Interface for manually adding research sources"""

    def __init__(self, task_queue: TaskQueue, discovery: SourceDiscovery):
        self.task_queue = task_queue
        self.discovery = discovery

    async def add_url_source(self, url: str, metadata: dict = None) -> str:
        """Add URL as research source and create task"""

        logger.info(f"Adding URL source: {url}")

        if not self._validate_url(url):
            raise ValueError(f"Invalid URL: {url}")

        source_id = await self.discovery.add_manual_source(url, metadata)

        task_id = await self.task_queue.add_task(
            task_type="FETCH",
            source=url,
            metadata={
                "source_id": source_id,
                "source_type": "url",
                "added_by": "manual",
                **(metadata or {}),
            },
        )

        logger.info(f"Created task {task_id} for URL {url}")
        return task_id

    async def add_file_source(self, file_path: str, metadata: dict = None) -> str:
        """Add file as research source and create task"""

        logger.info(f"Adding file source: {file_path}")

        if not Path(file_path).exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        wsl_path = file_path.replace("\\\\", "\\")

        source_id = await self.discovery.add_manual_source(file_path, metadata or {})

        task_id = await self.task_queue.add_task(
            task_type="FETCH",
            source=file_path,
            metadata={
                "source_id": source_id,
                "source_type": "file",
                "wsl_path": wsl_path,
                "added_by": "manual",
                **(metadata or {}),
            },
        )

        logger.info(f"Created task {task_id} for file {file_path}")
        return task_id

    async def add_text_source(self, text: str, metadata: dict = None) -> str:
        """Add text directly as research source"""

        logger.info(f"Adding text source ({len(text)} chars)")

        source_id = await self.discovery.add_manual_source(
            f"text_source_{len(text)}", metadata or {}
        )

        task_id = await self.task_queue.add_task(
            task_type="INGEST",
            source="direct_text",
            metadata={
                "source_id": source_id,
                "source_type": "text",
                "text_content": text,
                "added_by": "manual",
                **(metadata or {}),
            },
        )

        logger.info(f"Created task {task_id} for direct text")
        return task_id

    def _validate_url(self, url: str) -> bool:
        """Validate URL format"""

        # Simple validation - just check it starts with http/https and has basic structure
        if not url.startswith(("http://", "https://")):
            return False

        # Check for at least one dot in domain part
        try:
            from urllib.parse import urlparse

            parsed = urlparse(url)
            return bool(parsed.netloc and "." in parsed.netloc)
        except:
            return False
