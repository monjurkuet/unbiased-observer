import os
import json
import logging
import asyncio
from typing import List, Optional, Any
from pydantic import BaseModel, Field
from openai import AsyncOpenAI
import instructor
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# --- Pydantic Models for Structured Output ---

class Entity(BaseModel):
    name: str = Field(..., description="Unique name of the entity.")
    type: str = Field(..., description="Type of the entity (e.g., Person, Organization, Event, Concept).")
    description: str = Field(..., description="Comprehensive description of the entity based on the text.")

class Relationship(BaseModel):
    source: str = Field(..., description="Name of the source entity.")
    target: str = Field(..., description="Name of the target entity.")
    type: str = Field(..., description="Type of relationship (e.g., AUTHORED, LEADS, PART_OF). UPPERCASE.")
    description: str = Field(..., description="Contextual explanation of why this relationship exists.")
    weight: float = Field(default=1.0, description="Strength of relationship (0.0 to 1.0).")

class Event(BaseModel):
    primary_entity: str = Field(..., description="The main entity involved in the event.")
    description: str = Field(..., description="Specific description of what happened.")
    raw_time: str = Field(..., description="The original time description from the text (e.g., 'Q1 2026', 'last week').")
    normalized_date: Optional[str] = Field(None, description="ISO 8601 date if possible (YYYY-MM-DD).")

class KnowledgeGraph(BaseModel):
    """
    Structured representation of a knowledge graph extracted from text.
    """
    entities: List[Entity] = Field(default_factory=list)
    relationships: List[Relationship] = Field(default_factory=list)
    events: List[Event] = Field(default_factory=list)

# --- Ingestor Class ---


class GraphIngestor:
    def __init__(
        self,
        base_url: str = "http://localhost:8317/v1",
        api_key: str = "lm-studio",
        model_name: str = "gemini-2.5-pro", # Default to a strong model from your list
        mode: instructor.Mode = instructor.Mode.TOOLS
    ):
        """
        Initialize the Graph Ingestor with OpenAI-compatible local API.
        
        Args:
            base_url: The local endpoint for your inference server.
            api_key: API key (usually dummy for local servers).
            model_name: The ID of the model from your available list.
            mode: Instructor mode. Mode.TOOLS is recommended for frontier models.
        """
        self.raw_client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
        )
        self.client = instructor.from_openai(self.raw_client, mode=mode)
        self.model_name = model_name
        logger.info(f"Initialized GraphIngestor with model: {self.model_name}")

    async def list_available_models(self) -> List[str]:
        """Fetch the list of model IDs available on the server."""
        try:
            models = await self.raw_client.models.list()
            return [m.id for m in models.data]
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return []

    async def extract(self, text: str) -> KnowledgeGraph:
        """
        High-Fidelity Extraction Pipeline (2-Pass Gleaning).
        Uses hierarchical prompting to ensure zero compromise on quality.
        """
        logger.info(f"Starting extraction using model: {self.model_name}...")
        
        # Pass 1: Core Extraction
        core_graph = await self._pass_1_core(text)
        logger.info(f"Pass 1 complete. Found {len(core_graph.entities)} entities.")

        # Pass 2: Gleaning (Finding missed details)
        gleaned_graph = await self._pass_2_gleaning(text, core_graph)
        logger.info(f"Pass 2 complete. Found {len(gleaned_graph.entities)} additional entities.")

        # Merge
        final_graph = self._merge_graphs(core_graph, gleaned_graph)
        logger.info(f"Extraction complete. Final count: {len(final_graph.entities)} entities, {len(final_graph.relationships)} relationships.")
        
        return final_graph

    async def _pass_1_core(self, text: str) -> KnowledgeGraph:
        return await self.client.chat.completions.create(
            model=self.model_name,
            response_model=KnowledgeGraph,
            messages=[
                {
                    "role": "system", 
                    "content": "You are a Senior Knowledge Graph Architect and Historian. Your task is to extract a comprehensive, high-fidelity graph and TIMELINE from unstructured text."
                },
                {
                    "role": "user", 
                    "content": f"""
                    EXTRACT all significant entities, relationships, and CHRONOLOGICAL EVENTS.
                    
                    **Guidelines:**
                    1. ENTITIES: Identify People, Organizations, Projects, Concepts, and Locations.
                    2. RELATIONSHIPS: Define explicit, typed relationships in UPPERCASE.
                    3. EVENTS: Extract specific occurrences with their original time descriptions. 
                       - Link the event to its primary entity.
                       - Try to normalize the date to ISO 8601 (YYYY-MM-DD) if possible.
                    4. DESCRIPTIONS: Provide rich, factual descriptions for every node, edge, and event.
                    
                    **Text to Analyze:**
                    {text}
                    """
                }
            ],
            max_retries=3,
        )

    async def _pass_2_gleaning(self, text: str, existing_graph: KnowledgeGraph) -> KnowledgeGraph:
        """
        The 'Zero Compromise' quality pass. Finds details missed in the first pass.
        """
        existing_names = [e.name for e in existing_graph.entities]
        
        return await self.client.chat.completions.create(
            model=self.model_name,
            response_model=KnowledgeGraph,
            messages=[
                {
                    "role": "system", 
                    "content": "You are a Detail-Oriented Forensic Auditor. Your goal is to find missed entities, subtle relationships, and overlooked TEMPORAL EVENTS."
                },
                {
                    "role": "user", 
                    "content": f"""
                    I have already extracted these entities: {json.dumps(existing_names[:60])}.
                    
                    **Your Goal:**
                    Perform a second pass on the text. Identify:
                    1. ANY entity or relationship not listed above.
                    2. Specific DATES or TIME-BOUND milestones that were skipped.
                    3. Chronological links between events.
                    
                    **Constraint:** Only output NEW information.
                    
                    **Text to Analyze:**
                    {text}
                    """
                }
            ],
            max_retries=3,
        )

    def _merge_graphs(self, g1: KnowledgeGraph, g2: KnowledgeGraph) -> KnowledgeGraph:
        """
        Merge two graphs, preventing exact name duplicates.
        """
        entities = {e.name.lower(): e for e in g1.entities}
        
        for e in g2.entities:
            if e.name.lower() not in entities:
                entities[e.name.lower()] = e
            else:
                if len(e.description) > len(entities[e.name.lower()].description):
                    entities[e.name.lower()].description = e.description
                
        # Merge edges
        edges = set()
        final_edges = []
        
        for r in g1.relationships + g2.relationships:
            key = (r.source.lower(), r.target.lower(), r.type.upper())
            if key not in edges:
                edges.add(key)
                final_edges.append(r)
        
        # Merge events (simple dedupe by description)
        event_descs = set()
        final_events = []
        for ev in g1.events + g2.events:
            if ev.description not in event_descs:
                event_descs.add(ev.description)
                final_events.append(ev)
                
        return KnowledgeGraph(entities=list(entities.values()), relationships=final_edges, events=final_events)

# --- Usage Example ---
if __name__ == "__main__":
    async def run_test():
        # You can pick any model from your list here
        ingestor = GraphIngestor(model_name="gemini-2.5-pro") 
        
        sample_text = """
        Project Alpha is a confidential research initiative led by Dr. Sarah Chen at the AI Research Division of Cyberdyne Systems. 
        It focuses on developing self-optimizing cognitive architectures. 
        The project received funding from the Department of Advanced Technology in late 2024.
        """
        
        try:
            # Check models first to be sure
            available = await ingestor.list_available_models()
            print(f"Available Models: {available}")
            
            if ingestor.model_name not in available and available:
                print(f"Warning: {ingestor.model_name} not found. Switching to {available[0]}")
                ingestor.model_name = available[0]

            result = await ingestor.extract(sample_text)
            print("\n--- Extracted Knowledge Graph ---")
            print(result.model_dump_json(indent=2))
            
        except Exception as e:
            logger.error(f"Extraction failed: {e}")

    asyncio.run(run_test())
