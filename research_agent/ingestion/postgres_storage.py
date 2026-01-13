import psycopg
from psycopg import sql
from typing import List
import logging
import os

logger = logging.getLogger("research_agent.ingestion")


class DirectPostgresStorage:
    """Direct PostgreSQL access for knowledge storage"""

    def __init__(self, config):
        self.conn_str = config.database.connection_string
        self.pool = None
        self.embedding_api_key = os.getenv("GOOGLE_API_KEY")

    async def initialize(self):
        """Initialize connection pool"""

        if self.pool is None:
            self.pool = await psycopg.AsyncConnectionPool(
                self.conn_str, min_size=5, max_size=20
            )
            logger.info("PostgreSQL storage initialized")

    async def close(self):
        """Close connection pool"""

        if self.pool:
            await self.pool.close()
            logger.info("PostgreSQL storage closed")

    async def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding using Google GenAI"""

        if not self.embedding_api_key:
            logger.warning("GOOGLE_API_KEY missing, using dummy embedding")
            return [0.0] * 768

        try:
            from google.generativeai import genai

            genai.configure(api_key=self.embedding_api_key)
            result = genai.embed_content(
                model="models/text-embedding-004",
                content=text,
                task_type="retrieval_document",
            )
            return result["embedding"]
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}", exc_info=True)
            return [0.0] * 768

    async def store_entity(
        self,
        name: str,
        entity_type: str,
        description: str,
        embedding: List[float] = None,
    ) -> str:
        """Store entity directly, return UUID"""

        if embedding is None:
            embedding = await self.generate_embedding(f"{name} {description}")

        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    INSERT INTO nodes (name, type, description, embedding)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (name, type)
                    DO UPDATE SET description = EXCLUDED.description,
                                    embedding = EXCLUDED.embedding,
                                    updated_at = NOW()
                    RETURNING id
                    """,
                    (name, entity_type, description, embedding),
                )
                row = await cur.fetchone()
                if row is None:
                    await cur.execute(
                        "SELECT id FROM nodes WHERE name = %s AND type = %s",
                        (name, entity_type),
                    )
                    row = await cur.fetchone()

        entity_id = str(row[0])
        logger.debug(f"Stored entity: {entity_id} ({name})")
        return entity_id

    async def store_entities_batch(self, entities: List[dict]) -> dict:
        """Store multiple entities in batch"""

        logger.info(f"Storing {len(entities)} entities in batch")

        entity_id_map = {}

        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                query = sql.SQL("""
                    INSERT INTO nodes (name, type, description, embedding)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (name, type)
                    DO UPDATE SET description = EXCLUDED.description,
                                    embedding = EXCLUDED.embedding,
                                    updated_at = NOW()
                    RETURNING id
                """)

                data = []
                for entity in entities:
                    embedding = entity.get("embedding")
                    if embedding is None:
                        embedding = await self.generate_embedding(
                            f"{entity['name']} {entity['description']}"
                        )
                    data.append(
                        (
                            entity["name"],
                            entity["type"],
                            entity["description"],
                            embedding,
                        )
                    )

                await cur.executemany(query, data)

                for i, (row,) in enumerate(await cur):
                    entity_id = str(row[0])
                    entity_id_map[entities[i]["name"]] = entity_id

        logger.info(f"Batch stored {len(entity_id_map)} entities")
        return entity_id_map

    async def store_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: str,
        description: str = None,
        weight: float = 1.0,
    ) -> bool:
        """Store edge directly"""

        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    INSERT INTO edges (source_id, target_id, type, description, weight)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (source_id, target_id, type) DO NOTHING
                    """,
                    (source_id, target_id, edge_type, description, weight),
                )

        logger.debug(f"Stored edge: {source_id} --[{edge_type}]--> {target_id}")
        return True

    async def store_edges_batch(self, edges: List[dict], entity_id_map: dict) -> int:
        """Store multiple edges in batch"""

        logger.info(f"Storing {len(edges)} edges in batch")

        stored = 0

        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                query = sql.SQL("""
                    INSERT INTO edges (source_id, target_id, type, description, weight)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (source_id, target_id, type) DO NOTHING
                """)

                data = []
                for edge in edges:
                    source_id = entity_id_map.get(edge["source"])
                    target_id = entity_id_map.get(edge["target"])

                    if source_id and target_id:
                        data.append(
                            (
                                source_id,
                                target_id,
                                edge["type"],
                                edge.get("description"),
                                edge.get("weight", 1.0),
                            )
                        )

                if data:
                    await cur.executemany(query, data)
                    stored = len(data)

        logger.info(f"Batch stored {stored} edges")
        return stored

    async def store_event(
        self,
        node_id: str,
        description: str,
        timestamp: str = None,
        raw_time: str = None,
    ) -> bool:
        """Store event directly"""

        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    INSERT INTO events (node_id, description, timestamp, raw_time_desc)
                    VALUES (%s, %s, %s, %s)
                    """,
                    (node_id, description, timestamp, raw_time),
                )

        logger.debug(f"Stored event: {node_id}")
        return True

    async def store_events_batch(self, events: List[dict], entity_id_map: dict) -> int:
        """Store multiple events in batch"""

        logger.info(f"Storing {len(events)} events in batch")

        stored = 0

        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                query = sql.SQL("""
                    INSERT INTO events (node_id, description, timestamp, raw_time_desc)
                    VALUES (%s, %s, %s, %s)
                """)

                data = []
                for event in events:
                    node_id = entity_id_map.get(event["primary_entity"])

                    if node_id:
                        data.append(
                            (
                                node_id,
                                event["description"],
                                event.get("normalized_date"),
                                event.get("raw_time"),
                            )
                        )

                if data:
                    await cur.executemany(query, data)
                    stored = len(data)

        logger.info(f"Batch stored {stored} events")
        return stored
