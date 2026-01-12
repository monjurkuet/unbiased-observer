from langflow.custom import Component
from langflow.io import Output, SecretStrInput, StrInput, IntInput, HandleInput
from langflow.field_typing import Tool, Embeddings
from langchain.tools import StructuredTool
from pydantic import BaseModel, Field
import psycopg
from psycopg.rows import dict_row
from typing import Optional, List, Dict, Any
import logging

class GlobalKnowledgeTool(Component):
    display_name = "Global Knowledge Base Tool"
    description = "Query hierarchical communities, timelines, and entity graphs from the High-Fidelity KB."
    icon = "database"
    name = "GlobalKnowledgeTool"

    inputs = [
        SecretStrInput(
            name="connection_string",
            display_name="Database Connection String",
            info="PostgreSQL connection string (postgresql://user:pass@host:port/db)",
            required=True
        ),
        HandleInput(
            name="embedding_model",
            display_name="Embedding Model",
            input_types=["Embeddings"],
            info="Connect a Google or OpenAI Embeddings component to enable semantic search.",
            required=True
        ),
        IntInput(
            name="result_limit",
            display_name="Result Limit",
            value=5,
            info="Max records to return per search."
        )
    ]

    outputs = [
        Output(display_name="Knowledge Tool", name="tool", method="build_tool"),
    ]

    def build_tool(self) -> Tool:
        """
        Builds a StructuredTool that the Agent can call.
        """
        return StructuredTool.from_function(
            name="query_knowledge_base",
            description="""
            Useful for querying the Knowledge Base across three levels:
            1. 'global': To get high-level thematic summaries and executive reports.
            2. 'timeline': To see a chronological list of events for an entity or project.
            3. 'local': To get detailed information about a specific entity and its relationships.
            
            Input: A query string and a search_mode ('global', 'timeline', or 'local').
            """,
            func=self.execute_query,
        )

    def execute_query(self, query: str, search_mode: str = "global") -> str:
        """
        Orchestrates the retrieval based on mode.
        """
        try:
            if search_mode == "global":
                return self._search_communities(query)
            elif search_mode == "timeline":
                return self._search_timeline(query)
            elif search_mode == "local":
                return self._search_entities(query)
            else:
                return f"Error: Invalid search_mode '{search_mode}'. Use 'global', 'timeline', or 'local'."
        except Exception as e:
            return f"Error executing knowledge query: {str(e)}"

    def _search_communities(self, query: str) -> str:
        """
        Semantic search over community summaries.
        """
        # 1. Embed the query
        query_vector = self.embedding_model.embed_query(query)
        
        results = []
        with psycopg.connect(self.connection_string, row_factory=dict_row) as conn:
            with conn.cursor() as cur:
                # Use pgvector cosine similarity (<=>)
                cur.execute(
                    """
                    SELECT title, summary, level, 1 - (embedding <=> %s::vector) as similarity
                    FROM communities
                    WHERE embedding IS NOT NULL
                    ORDER BY similarity DESC
                    LIMIT %s
                    """,
                    (query_vector, self.result_limit)
                )
                for row in cur:
                    results.append(
                        f"### {row['title']} (Level {row['level']})\n"
                        f"RELEVANCE: {row['similarity']:.2f}\n"
                        f"SUMMARY: {row['summary']}\n"
                    )
        
        return "\n".join(results) if results else "No high-level community reports found matching your query."

    def _search_timeline(self, query: str) -> str:
        """
        Keyword search over the events table.
        """
        results = []
        with psycopg.connect(self.connection_string, row_factory=dict_row) as conn:
            with conn.cursor() as cur:
                # Basic ILIKE search for now, could be upgraded to Trigram/FTS
                cur.execute(
                    """
                    SELECT n.name as entity, e.description, e.raw_time_desc, e.normalized_date
                    FROM events e
                    JOIN nodes n ON e.node_id = n.id
                    WHERE e.description ILIKE %s OR n.name ILIKE %s
                    ORDER BY e.normalized_date ASC NULLS LAST, e.created_at ASC
                    LIMIT %s
                    """,
                    (f"%{query}%", f"%{query}%", self.result_limit * 2)
                )
                for row in cur:
                    time = row['raw_time_desc'] or row['normalized_date'] or "Unknown Time"
                    results.append(f"- [{time}] {row['entity']}: {row['description']}")
        
        return "CHRONOLOGICAL TIMELINE:\n" + "\n".join(results) if results else "No events found matching your query."

    def _search_entities(self, query: str) -> str:
        """
        Local search: Entity details + outgoing relationships.
        """
        results = []
        with psycopg.connect(self.connection_string, row_factory=dict_row) as conn:
            with conn.cursor() as cur:
                # 1. Find the best matching node (Fuzzy search)
                cur.execute(
                    """
                    SELECT id, name, type, description 
                    FROM nodes 
                    WHERE name % %s OR description ILIKE %s
                    ORDER BY similarity(name, %s) DESC
                    LIMIT 1
                    """,
                    (query, f"%{query}%", query)
                )
                node = cur.fetchone()
                
                if not node:
                    return "No specific entity found matching that name."
                
                results.append(f"ENTITY: {node['name']} ({node['type']})")
                results.append(f"DESCRIPTION: {node['description']}")
                
                # 2. Get relationships
                cur.execute(
                    """
                    SELECT e.type, n.name as target, e.description
                    FROM edges e
                    JOIN nodes n ON e.target_id = n.id
                    WHERE e.source_id = %s
                    """,
                    (node['id'],)
                )
                rels = cur.fetchall()
                if rels:
                    results.append("\nRELATIONSHIPS:")
                    for r in rels:
                        results.append(f"- {r['type']} -> {r['target']} ({r['description']})")
                        
        return "\n".join(results)
