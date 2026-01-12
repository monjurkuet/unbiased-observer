import asyncio
import logging
import os
import argparse
from typing import List
from dotenv import load_dotenv

from ingestor import GraphIngestor, KnowledgeGraph
from resolver import EntityResolver
from community import CommunityDetector

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

class KnowledgePipeline:
    def __init__(self):
        # Configuration
        self.db_conn_str = f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
        
        # Initialize Components
        self.ingestor = GraphIngestor(model_name="gemini-2.5-flash") # Default strong model
        self.resolver = EntityResolver(db_conn_str=self.db_conn_str, model_name="gemini-2.5-flash")
        self.community_detector = CommunityDetector(db_conn_str=self.db_conn_str)

    async def run(self, file_path: str):
        """
        Run the full High-Fidelity Pipeline on a single file.
        """
        logger.info(f"=== Starting Pipeline for {file_path} ===")
        
        # 1. Read File
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
        except Exception as e:
            logger.error(f"Failed to read file: {e}")
            return

        # 2. Ingestion (Extraction)
        logger.info("--- Stage 1: High-Resolution Extraction ---")
        graph: KnowledgeGraph = await self.ingestor.extract(text)
        
        # 3. Resolution & Storage
        logger.info("--- Stage 2: Hybrid Entity Resolution & Storage ---")
        await self._store_graph(graph)

        # 4. Community Detection (Optional: Usually run in batch, not per file)
        # We'll run it here for the demo/prototype feel
        logger.info("--- Stage 3: Community Detection ---")
        G = await self.community_detector.load_graph()
        if G.number_of_nodes() > 0:
            memberships = self.community_detector.detect_communities(G)
            await self.community_detector.save_communities(memberships)
        
        # 5. Recursive Summarization
        logger.info("--- Stage 4: Recursive Summarization ---")
        from summarizer import CommunitySummarizer
        summarizer = CommunitySummarizer(self.db_conn_str, model_name="gemini-2.5-flash")
        await summarizer.summarize_all()
        
        logger.info("=== Pipeline Complete ===")

    async def _store_graph(self, graph: KnowledgeGraph):
        """
        Resolves entities and inserts edges.
        """
        # Map localized entity names to resolved DB UUIDs
        # name_to_id = {"Project Alpha": "uuid-123", ...}
        name_to_id = {}
        
        # 1. Resolve Entities
        for entity in graph.entities:
            # TODO: Generate embedding for entity (using a simplified method or call an embedding service)
            # For this prototype, we'll use a placeholder or call Google Embeddings if available
            embedding = await self._get_embedding(f"{entity.name} {entity.description}")
            
            # Resolve and Insert
            # Convert Pydantic model to dict for resolver
            entity_dict = entity.model_dump()
            resolved_id = await self.resolver.resolve_and_insert(entity_dict, embedding)
            name_to_id[entity.name] = resolved_id
            
        # 2. Insert Edges
        # We need a direct DB connection here or move this to Resolver
        from psycopg import AsyncConnection
        async with await AsyncConnection.connect(self.db_conn_str) as conn:
            async with conn.cursor() as cur:
                for edge in graph.relationships:
                    source_id = name_to_id.get(edge.source)
                    target_id = name_to_id.get(edge.target)
                    
                    if source_id and target_id:
                        await cur.execute(
                            """
                            INSERT INTO edges (source_id, target_id, type, description, weight)
                            VALUES (%s, %s, %s, %s, %s)
                            ON CONFLICT (source_id, target_id, type) DO NOTHING
                            """,
                            (source_id, target_id, edge.type, edge.description, edge.weight)
                        )
                
                # 3. Insert Events
                for event in graph.events:
                    node_id = name_to_id.get(event.primary_entity)
                    if node_id:
                        # Normalize date
                        clean_date = event.normalized_date
                        if clean_date and len(clean_date) == 4 and clean_date.isdigit():
                            clean_date = f"{clean_date}-01-01"
                        elif clean_date and len(clean_date) == 7 and clean_date[4] == '-':
                            clean_date = f"{clean_date}-01" # Handle YYYY-MM
                        
                        # Use a sub-transaction (SAVEPOINT) to protect the main transaction
                        try:
                            async with conn.transaction():
                                await cur.execute(
                                    """
                                    INSERT INTO events (node_id, description, timestamp, raw_time_desc)
                                    VALUES (%s, %s, %s, %s)
                                    """,
                                    (node_id, event.description, clean_date, event.raw_time)
                                )
                        except Exception as e:
                            logger.warning(f"Skipping invalid date '{clean_date}' for event: {event.description}. Error: {e}")
                            # Try one more time without the date
                            try:
                                async with conn.transaction():
                                    await cur.execute(
                                        """
                                        INSERT INTO events (node_id, description, raw_time_desc)
                                        VALUES (%s, %s, %s)
                                        """,
                                        (node_id, event.description, event.raw_time)
                                    )
                            except Exception:
                                pass # Give up on this specific event
                
                await conn.commit()

    async def _get_embedding(self, text: str) -> List[float]:
        """
        Helper to get embeddings. 
        In production, inject an EmbeddingService class.
        For now, uses Google GenAI if available, else random/zero for testing.
        """
        import google.generativeai as genai
        api_key = os.getenv("GOOGLE_API_KEY")
        if api_key:
            genai.configure(api_key=api_key)
            result = genai.embed_content(
                model="models/text-embedding-004",
                content=text,
                task_type="retrieval_document"
            )
            return result['embedding']
        else:
            logger.warning("GOOGLE_API_KEY missing. Using dummy embedding.")
            return [0.0] * 768

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Knowledge Base Pipeline")
    parser.add_argument("file", help="Path to text file to ingest")
    args = parser.parse_args()

    pipeline = KnowledgePipeline()
    asyncio.run(pipeline.run(args.file))
