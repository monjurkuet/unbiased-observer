import uuid
import yaml
from pathlib import Path
import logging
from typing import List, Dict

logger = logging.getLogger("research_agent.research")


class SourceDiscovery:
    """Discover and manage research sources"""

    def __init__(self, sources_config: str):
        self.sources_config = sources_config
        self.sources = self._load_sources(sources_config)

    def _load_sources(self, config_path: str) -> List[Dict]:
        """Load configured research sources from YAML"""

        config_file = Path(config_path)
        if not config_file.exists():
            logger.warning(f"Config file not found: {config_path}")
            return [{"type": "manual", "name": "Manual Input", "active": True}]

        with open(config_file, "r") as f:
            config_data = yaml.safe_load(f)

        sources = config_data.get("sources", [])
        logger.info(f"Loaded {len(sources)} research sources")

        return sources

    async def discover_new_content(self) -> List[Dict]:
        """Check for new content from configured sources"""

        logger.info("Discovering new content from sources...")

        discovered = []

        for source in self.sources:
            if not source.get("active", True):
                logger.debug(f"Skipping inactive source: {source.get('name')}")
                continue

            if source["type"] == "manual":
                logger.debug("Manual source - no auto-discovery")
                continue

            logger.warning(f"Source type {source['type']} not yet implemented")

        return discovered

    async def add_manual_source(self, url: str, metadata: Dict = None) -> str:
        """Add manual research source to database"""

        source_id = str(uuid.uuid4())

        logger.info(f"Added manual source: {source_id}")

        return source_id

    def get_sources(self) -> List[Dict]:
        """Get all configured sources"""
        return self.sources

    def get_active_sources(self) -> List[Dict]:
        """Get only active sources"""
        return [s for s in self.sources if s.get("active", True)]
