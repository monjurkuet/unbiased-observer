import asyncio
from research_agent.ingestion.async_ingestor import AsyncIngestor
from research_agent.ingestion.postgres_storage import DirectPostgresStorage
from knowledge_base.ingestor import KnowledgeGraph
import logging
from typing import Dict

logger = logging.getLogger("research_agent.ingestion")


class IngestionPipeline:
    """Full ingestion pipeline coordination"""

    def __init__(self, config):
        self.config = config
        self.ingestor = AsyncIngestor(
            base_url=config.llm.base_url,
            api_key=config.llm.api_key,
            model_name=config.llm.model_default,
        )
        self.storage = DirectPostgresStorage(config)

    async def initialize(self):
        """Initialize components"""

        await self.storage.initialize()
        logger.info("Ingestion pipeline initialized")

    async def ingest_content(self, content: str, metadata: Dict = None) -> Dict:
        """Full ingestion pipeline from text content"""

        source = metadata.get("source", "unknown") if metadata else "unknown"
        logger.info(f"Starting ingestion: {source}")

        start_time = asyncio.get_event_loop().time()

        try:
            logger.info("Stage 1: Extracting knowledge graph...")
            graph = await self.ingestor.extract_async(content)

            logger.info("Stage 2: Storing entities...")
            entity_data = [
                {"name": e.name, "type": e.type, "description": e.description}
                for e in graph.entities
            ]

            entity_id_map = await self.storage.store_entities_batch(entity_data)

            logger.info("Stage 3: Storing edges...")
            edge_data = [
                {
                    "source": r.source,
                    "target": r.target,
                    "type": r.type,
                    "description": r.description,
                    "weight": r.weight,
                }
                for r in graph.relationships
            ]

            edges_stored = await self.storage.store_edges_batch(
                edge_data, entity_id_map
            )

            logger.info("Stage 4: Storing events...")
            event_data = [
                {
                    "primary_entity": e.primary_entity,
                    "description": e.description,
                    "normalized_date": e.normalized_date,
                    "raw_time": e.raw_time,
                }
                for e in graph.events
            ]

            events_stored = await self.storage.store_events_batch(
                event_data, entity_id_map
            )

            end_time = asyncio.get_event_loop().time()
            duration = end_time - start_time

            result = {
                "status": "completed",
                "entities_stored": len(graph.entities),
                "edges_stored": edges_stored,
                "events_stored": events_stored,
                "duration_seconds": duration,
                "source": source,
            }

            logger.info(
                f"Ingestion complete: "
                f"{len(graph.entities)} entities, "
                f"{edges_stored} edges, "
                f"{events_stored} events, "
                f"duration: {duration:.2f}s"
            )

            return result

        except Exception as e:
            logger.error(f"Ingestion failed: {e}", exc_info=True)
            return {"status": "failed", "error": str(e), "source": source}

    async def ingest_from_fetch_result(self, fetch_result: Dict) -> Dict:
        """Ingest content from fetch result"""

        content = fetch_result.get("content")
        metadata = fetch_result.get("metadata", {})

        if not content:
            logger.warning("No content to ingest")
            return {"status": "skipped", "reason": "no_content"}

        return await self.ingest_content(content, metadata)
