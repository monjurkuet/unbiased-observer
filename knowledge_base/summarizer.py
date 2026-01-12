import logging
import asyncio
import json
import os
from typing import List, Dict, Optional
from pydantic import BaseModel, Field
import openai
import instructor
from psycopg import AsyncConnection
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

load_dotenv()

# --- Pydantic Models for Summaries ---


class Finding(BaseModel):
    summary: str = Field(..., description="Concise statement of a key fact or insight.")
    explanation: str = Field(..., description="Detailed explanation of the finding.")


class CommunityReport(BaseModel):
    title: str = Field(
        ..., description="A descriptive title for this community of entities."
    )
    summary: str = Field(
        ..., description="High-level executive summary of the community's themes."
    )
    rating: float = Field(
        ...,
        description="Importance rating (0-10) of this community to the overall domain.",
    )
    findings: List[Finding] = Field(
        ..., description="List of specific insights or claims found in this community."
    )


# --- Summarizer Class ---


class CommunitySummarizer:
    def __init__(
        self,
        db_conn_str: str,
        base_url: str = "http://localhost:8317/v1",
        api_key: str = "lm-studio",
        model_name: str = "gemini-2.5-pro",
    ):
        self.db_conn_str = db_conn_str
        self.raw_client = openai.OpenAI(base_url=base_url, api_key=api_key)
        self.client = instructor.from_openai(
            self.raw_client, mode=instructor.Mode.TOOLS
        )
        self.model_name = model_name

    async def summarize_all(self):
        """
        Main entry point: Summarize communities level by level, bottom-up.
        """
        # 1. Get max level
        async with await AsyncConnection.connect(self.db_conn_str) as conn:
            async with conn.cursor() as cur:
                await cur.execute("SELECT MAX(level) FROM communities")
                row = await cur.fetchone()
                max_level = row[0] if row and row[0] is not None else 0

        logger.info(f"Starting Recursive Summarization. Max Level: {max_level}")

        # 2. Iterate levels from 0 (Leaf) to Max (Root)
        # Note: In some implementations, 0 is root. In ours (Leiden), let's assume 0 is the finest grain (Leaf).
        # We process Level 0 first, then Level 1 can use Level 0's summaries.

        for level in range(max_level + 1):
            logger.info(f"--- Processing Level {level} ---")
            await self._process_level(level)

    async def _process_level(self, level: int):
        """
        Summarize all communities at a specific level.
        """
        async with await AsyncConnection.connect(self.db_conn_str) as conn:
            async with conn.cursor() as cur:
                # Fetch communities at this level needing summarization
                # (For now, we re-summarize everything. Production would use a 'dirty' flag)
                await cur.execute(
                    "SELECT id, title FROM communities WHERE level = %s", (level,)
                )
                communities = await cur.fetchall()

                for comm_id, title in communities:
                    await self._summarize_community(comm_id, level)

    async def _summarize_community(self, community_id: str, level: int):
        """
        Generate a report for a single community.
        """
        context_text = await self._gather_context(community_id, level)

        if not context_text:
            logger.warning(f"No context found for Community {community_id}. Skipping.")
            return

        logger.info(
            f"Generating report for Community {community_id} (Level {level})..."
        )

        try:
            report: CommunityReport = self.client.chat.completions.create(
                model=self.model_name,
                response_model=CommunityReport,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert Intelligence Analyst. Your goal is to synthesize structured data into a comprehensive report.",
                    },
                    {
                        "role": "user",
                        "content": f"""
                        **Task:** Analyze the provided entities and relationships to generate a Community Report.
                        
                        **Context (Entities & Relations):**
                        {context_text[:50000]}  # Truncate to avoid context overflow if huge
                        
                        **Requirements:**
                        1. Title: Create a specific, descriptive title.
                        2. Summary: Write a high-level overview.
                        3. Findings: List key insights, contradictions, or patterns.
                        4. Rating: Assess importance (0-10).
                        """,
                    },
                ],
                max_retries=3,
            )

            # Save to DB
            await self._save_report(community_id, report)

        except Exception as e:
            logger.error(f"Failed to summarize community {community_id}: {e}")

    async def _gather_context(self, community_id: str, level: int) -> str:
        """
        Gather text context.
        - If Level 0: Gather raw Entity descriptions + Relations.
        - If Level > 0: Gather Summaries of child communities (Recursive step).
        """
        context = []

        async with await AsyncConnection.connect(self.db_conn_str) as conn:
            async with conn.cursor() as cur:
                if level == 0:
                    # Gather Entities
                    await cur.execute(
                        """
                        SELECT n.name, n.description 
                        FROM community_membership cm
                        JOIN nodes n ON cm.node_id = n.id
                        WHERE cm.community_id = %s
                        LIMIT 50
                        """,
                        (community_id,),
                    )
                    rows = await cur.fetchall()
                    context.append("### Member Entities:")
                    for r in rows:
                        context.append(f"- {r[0]}: {r[1]}")

                    # Gather Relationships
                    await cur.execute(
                        """
                        SELECT n1.name, e.type, n2.name, e.description
                        FROM edges e
                        JOIN nodes n1 ON e.source_id = n1.id
                        JOIN nodes n2 ON e.target_id = n2.id
                        JOIN community_membership cm1 ON n1.id = cm1.node_id
                        JOIN community_membership cm2 ON n2.id = cm2.node_id
                        WHERE cm1.community_id = %s AND cm2.community_id = %s
                        LIMIT 50
                        """,
                        (community_id, community_id),
                    )
                    rows = await cur.fetchall()
                    context.append("\n### Internal Relationships:")
                    for r in rows:
                        context.append(f"- {r[0]} --[{r[1]}]--> {r[2]}: {r[3]}")

                else:
                    # Level > 0: Gather Child Community Summaries
                    await cur.execute(
                        """
                        SELECT c.title, c.summary 
                        FROM community_hierarchy ch
                        JOIN communities c ON ch.child_id = c.id
                        WHERE ch.parent_id = %s
                        """,
                        (community_id,),
                    )
                    rows = await cur.fetchall()
                    context.append(f"### Sub-Communities (Children):")
                    for r in rows:
                        context.append(f"#### {r[0]}\nSummary: {r[1]}\n")

        return "\n".join(context)

    async def _save_report(self, community_id: str, report: CommunityReport):
        """
        Persist the generated report and its vector embedding.
        """
        # Generate embedding for the summary to enable global semantic search
        embedding = await self._get_embedding(f"{report.title} {report.summary}")

        async with await AsyncConnection.connect(self.db_conn_str) as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    UPDATE communities 
                    SET title = %s, summary = %s, full_content = %s, embedding = %s, updated_at = NOW()
                    WHERE id = %s
                    """,
                    (
                        report.title,
                        report.summary,
                        report.model_dump_json(),
                        embedding,
                        community_id,
                    ),
                )
                await conn.commit()

    async def _get_embedding(self, text: str) -> List[float]:
        """
        Helper to get embeddings using Google GenAI.
        """
        import google.generativeai as genai

        api_key = os.getenv("GOOGLE_API_KEY")
        if api_key:
            genai.configure(api_key=api_key)
            result = genai.embed_content(
                model="models/text-embedding-004",
                content=text,
                task_type="retrieval_document",
            )
            return result["embedding"]
        else:
            logger.warning(
                "GOOGLE_API_KEY missing for summarizer. Using dummy embedding."
            )
            return [0.0] * 768


# --- CLI Test ---
if __name__ == "__main__":

    async def main():
        import os

        conn_str = f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"

        summarizer = CommunitySummarizer(conn_str, model_name="gemini-2.5-pro")
        await summarizer.summarize_all()

    asyncio.run(main())
