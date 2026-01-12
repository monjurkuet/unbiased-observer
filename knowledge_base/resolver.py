import logging
import json
import asyncio
from typing import List, Dict, Tuple, Optional
from pydantic import BaseModel, Field
import openai
import instructor
from psycopg import AsyncConnection
import psycopg
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# --- Resolution Decision Model ---


class ResolutionDecision(BaseModel):
    decision: str = Field(
        ...,
        description="One of: 'MERGE' (same entity), 'LINK' (related but distinct), 'KEEP_SEPARATE' (unrelated).",
    )
    reasoning: str = Field(
        ..., description="Brief explanation of why this decision was made."
    )
    canonical_name: Optional[str] = Field(
        None, description="If MERGE, the best name to use for the unified entity."
    )


# --- Entity Resolver Class ---


class EntityResolver:
    def __init__(
        self,
        db_conn_str: str,
        base_url: str = "http://localhost:8317/v1",
        api_key: str = "lm-studio",
        model_name: str = "gemini-2.5-pro",
    ):
        """
        Hybrid Entity Resolver: Combines Vector Similarity (Recall) with LLM Reasoning (Precision).
        """
        self.db_conn_str = db_conn_str
        self.raw_client = openai.OpenAI(
            base_url=base_url,
            api_key=api_key,
        )
        self.client = instructor.from_openai(
            self.raw_client, mode=instructor.Mode.TOOLS
        )
        self.model_name = model_name

    async def find_candidates(
        self, entity_name: str, embedding: List[float], threshold: float = 0.70
    ) -> List[Dict]:
        """
        Find potential duplicates in the DB using vector similarity.
        """
        candidates = []
        async with await psycopg.AsyncConnection.connect(self.db_conn_str) as conn:
            async with conn.cursor() as cur:
                # Use cosine distance (<=>). Lower is better, so 1 - distance = similarity.
                # Threshold check: distance < (1 - threshold)
                await cur.execute(
                    """
                    SELECT id, name, type, description, 1 - (embedding <=> %s::vector) as similarity
                    FROM nodes
                    WHERE 1 - (embedding <=> %s::vector) > %s
                    ORDER BY similarity DESC
                    LIMIT 5
                    """,
                    (embedding, embedding, threshold),
                )
                async for row in cur:
                    candidates.append(
                        {
                            "id": str(row[0]),
                            "name": row[1],
                            "type": row[2],
                            "description": row[3],
                            "similarity": row[4],
                        }
                    )
        return candidates

    def _normalize_name(self, name: str) -> str:
        """Normalize name by removing titles, parentheses, and extra whitespace."""
        import re

        # Remove titles
        normalized = re.sub(
            r"\b(Dr\.?|Director|Prof\.?|Mr\.?|Ms\.?|Mrs\.?)\s*",
            "",
            name,
            flags=re.IGNORECASE,
        )
        # Remove content in parentheses
        normalized = re.sub(r"\([^)]*\)", "", normalized)
        # Remove extra whitespace and quotes
        normalized = re.sub(r"\s+", " ", normalized.strip())
        normalized = normalized.replace("'", "").replace('"', "")
        return normalized.lower()

    def judge_pair(
        self, new_entity: Dict, existing_candidate: Dict
    ) -> ResolutionDecision:
        """
        Ask the LLM to decide if two entities are the same.
        """
        return self.client.chat.completions.create(
            model=self.model_name,
            response_model=ResolutionDecision,
            messages=[
                {
                    "role": "system",
                    "content": f"""
                       **Task:** Compare these two entities and decide their relationship.
                       
                       **Entity A (New):**
                       - Name: {new_entity["name"]}
                       - Type: {new_entity["type"]}
                       - Desc: {new_entity.get("description", "N/A")}
                       
                       **Entity B (Existing in DB):**
                       - Name: {existing_candidate["name"]}
                       - Type: {existing_candidate["type"]}
                       - Desc: {existing_candidate["description"]}
                       
                       **Special Instructions for Person Entities:**
                       - Consider common name variations: titles (Dr., Director, Prof.), nicknames (Sam/Samuel), middle initials (J./John)
                       - Same person can have different roles/titles over time
                       - Focus on core identity: last name + first name root
                       
                       **Options:**
                       - MERGE: They are the same real-world entity (e.g., "Dr. Samuel Oakley" vs "Director Samuel Oakley")
                       - LINK: They are closely related but distinct (e.g., "Apple" vs "Apple iPhone")
                       - KEEP_SEPARATE: They are different or just share a generic name
                       
                       Make a decision based on whether these represent the same real-world entity.
                       """,
                },
            ],
            max_retries=3,
        )

    async def resolve_and_insert(self, entity: Dict, embedding: List[float]) -> str:
        candidates = await self.find_candidates(entity["name"], embedding)

        for candidate in candidates:
            if (
                candidate["name"].lower() == entity["name"].lower()
                and candidate["type"] == entity["type"]
            ):
                logger.info(
                    f"Exact match found for {entity['name']}. Returning existing ID."
                )
                return candidate["id"]

            normalized_new = self._normalize_name(entity["name"])
            normalized_candidate = self._normalize_name(candidate["name"])
            if (
                normalized_new == normalized_candidate
                and candidate["type"] == entity["type"]
            ):
                logger.info(
                    f"Normalized match found for {entity['name']}. Returning existing ID."
                )
                return candidate["id"]

            logger.info(
                f"Judging pair: {entity['name']} vs {candidate['name']} (Sim: {candidate['similarity']:.2f})"
            )
            decision = self.judge_pair(entity, candidate)

            if decision.decision == "MERGE":
                logger.info(
                    f"Merging {entity['name']} -> {candidate['name']} ({decision.reasoning})"
                )
                return candidate["id"]

            elif decision.decision == "LINK":
                pass

        return await self._insert_entity(entity, embedding)

    async def _insert_entity(self, entity: Dict, embedding: List[float]) -> str:
        async with await psycopg.AsyncConnection.connect(self.db_conn_str) as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    INSERT INTO nodes (name, type, description, embedding)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (name, type) DO UPDATE 
                    SET description = EXCLUDED.description, updated_at = NOW()
                    RETURNING id
                    """,
                    (
                        entity["name"],
                        entity["type"],
                        entity.get("description", ""),
                        embedding,
                    ),
                )
                row = await cur.fetchone()
                await conn.commit()
                if row is None:
                    await cur.execute(
                        "SELECT id FROM nodes WHERE name = %s AND type = %s",
                        (entity["name"], entity["type"]),
                    )
                    row = await cur.fetchone()
                    if row is None:
                        raise RuntimeError(
                            f"Failed to insert or find entity: {entity['name']}"
                        )
                return str(row[0])


# --- Usage Example ---
if __name__ == "__main__":
    # Dummy test without DB connection
    print("EntityResolver defined. Requires live DB and Embeddings to run full test.")
