import sys
import asyncio
from pathlib import Path
import logging
from knowledge_base.ingestor import GraphIngestor, KnowledgeGraph

logger = logging.getLogger("research_agent.ingestion")


class AsyncIngestor(GraphIngestor):
    """Async wrapper around existing GraphIngestor"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.semaphore = asyncio.Semaphore(3)

    async def extract_async(self, text: str) -> KnowledgeGraph:
        """Async extraction with concurrency control"""

        logger.info(f"Starting async extraction ({len(text)} chars)")

        async with self.semaphore:
            loop = asyncio.get_event_loop()
            graph = await loop.run_in_executor(None, lambda: super().extract(text))

        logger.info(
            f"Extraction complete: "
            f"{len(graph.entities)} entities, "
            f"{len(graph.relationships)} relationships, "
            f"{len(graph.events)} events"
        )

        return graph

    async def batch_extract_async(
        self, texts: list[str], batch_size: int = 10
    ) -> list[KnowledgeGraph]:
        """Batch extract multiple texts"""

        logger.info(f"Batch extracting {len(texts)} texts (batch_size={batch_size})")

        results = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            logger.info(
                f"Processing batch {i // batch_size + 1}/{(len(texts) - 1) // batch_size + 1}"
            )

            tasks = [self.extract_async(text) for text in batch]
            batch_results = await asyncio.gather(*tasks)

            results.extend(batch_results)

        logger.info(f"Batch extraction complete: {len(results)} graphs")
        return results
