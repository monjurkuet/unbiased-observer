# Knowledge Base (GraphRAG Engine)

This directory contains the high-fidelity **Agentic GraphRAG** ingestion and management system. It extends standard RAG by adding structured relationships, hierarchical community summaries, and chronological timelines.

## Features
- **High-Resolution Extraction:** Uses a 2-pass "gleaning" strategy with frontier models (via local OpenAI-compatible API) to capture subtle entities and relationships.
- **Hybrid Entity Resolution:** Combines vector similarity with LLM-based cognitive reasoning to deduplicate nodes (e.g., merging "Dr. Vance" and "Elena Vance").
- **Hierarchical Clustering:** Uses the **Leiden Algorithm** to cluster nodes into Micro-communities, which are then rolled up into Macro-themes.
- **Recursive Summarization:** Automatically generates "Intelligence Reports" for every community. Parent communities summarize their children, creating a searchable map of knowledge.
- **Temporal Tracking:** Extracts specific events and dates to build structured timelines.

## Core Components
- `pipeline.py`: The main orchestrator. Run this on any text file to ingest it.
- `ingestor.py`: The extraction engine (uses `instructor` + `openai`).
- `resolver.py`: The deduplication logic.
- `community.py`: The Leiden clustering implementation.
- `summarizer.py`: The recursive reporting engine.
- `visualize.py`: CLI tool to view the knowledge hierarchy as a tree.
- `langflow_tool.py`: A custom component for Langflow to give agents access to the KB.

## Quick Start

### 1. Requirements
Ensure your `.env` file in the root directory has the following:
- `DB_USER`, `DB_PASSWORD`, `DB_HOST`, `DB_PORT`, `DB_NAME`
- `GOOGLE_API_KEY` (for embeddings)
- `OPENAI_API_BASE` (pointing to your local http://localhost:8317/v1)

### 2. Ingest Data
```bash
uv run knowledge_base/pipeline.py path/to/your/data.txt
```

### 3. Visualize
```bash
uv run knowledge_base/visualize.py
```

## Database Schema
The system uses PostgreSQL with `pgvector`. See `schema.sql` for table definitions (`nodes`, `edges`, `communities`, `community_membership`, `community_hierarchy`, `events`).
