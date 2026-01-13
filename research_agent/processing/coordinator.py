import sys
import asyncio
from pathlib import Path
import logging
from knowledge_base.community import CommunityDetector
from knowledge_base.summarizer import CommunitySummarizer

logger = logging.getLogger("research_agent.processing")


class ProcessingCoordinator:
    """Coordinate community detection and summarization"""

    def __init__(self, config):
        self.config = config

        kb_path = Path(config.paths.knowledge_base)
        sys.path.insert(0, str(kb_path))

        self.community_detector = None
        self.summarizer = None

    async def initialize(self):
        """Initialize components"""

        self.community_detector = CommunityDetector(
            self.config.database.connection_string
        )
        self.summarizer = CommunitySummarizer(
            self.config.database.connection_string,
            base_url=self.config.llm.base_url,
            api_key=self.config.llm.api_key,
            model_name=self.config.llm.model_pro,
        )

        logger.info("Processing coordinator initialized")

    async def run_processing_pipeline(self) -> dict:
        """Run full processing pipeline"""

        logger.info("Starting processing pipeline")

        start_time = asyncio.get_event_loop().time()

        try:
            logger.info("Stage 1: Loading graph...")
            G = await self.community_detector.load_graph()

            if G.number_of_nodes() == 0:
                return {"status": "skipped", "reason": "empty_graph"}

            logger.info("Stage 2: Detecting communities...")
            memberships = self.community_detector.detect_communities(G)

            logger.info("Stage 3: Saving communities...")
            await self.community_detector.save_communities(memberships)

            logger.info("Stage 4: Summarizing communities...")
            await self.summarizer.summarize_all()

            end_time = asyncio.get_event_loop().time()
            duration = end_time - start_time

            unique_communities = len(set(m["cluster_id"] for m in memberships))

            result = {
                "status": "completed",
                "nodes_processed": G.number_of_nodes(),
                "communities_created": unique_communities,
                "duration_seconds": duration,
            }

            logger.info(
                f"Processing complete: "
                f"{G.number_of_nodes()} nodes, "
                f"{unique_communities} communities, "
                f"duration: {duration:.2f}s"
            )

            return result

        except Exception as e:
            logger.error(f"Processing failed: {e}", exc_info=True)
            return {"status": "failed", "error": str(e)}

    async def should_process(self) -> bool:
        """Check if processing should run"""

        from psycopg import AsyncConnection

        async with await AsyncConnection.connect(
            self.config.database.connection_string
        ) as conn:
            async with conn.cursor() as cur:
                await cur.execute("SELECT COUNT(*) FROM nodes")
                row = await cur.fetchone()
                entity_count = row[0]

        min_entities = self.config.processing.min_entities_to_process

        if entity_count < min_entities:
            logger.debug(
                f"Not enough entities to process: {entity_count} < {min_entities}"
            )
            return False

        return True
