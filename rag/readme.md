# Agentic Hybrid-Graph RAG

This directory contains the implementation of an **Agentic Hybrid-Graph Retrieval-Augmented Generation (RAG)** system. It allows autonomous agents to reason over knowledge by combining:
1.  **Vector Search (Semantic):** Finding information based on meaning using embeddings (Google `text-embedding-004`).
2.  **Graph Traversal (Structural):** Navigating explicit relationships between entities (e.g., "Who leads Project Alpha?").

## Architecture

The system uses **PostgreSQL** as a unified store for both the Knowledge Graph and Vector Embeddings.

*   **`nodes` Table:** Stores entities (People, Projects, Documents) with their 768-dimension vector embeddings.
*   **`edges` Table:** Stores directed, typed relationships between nodes.
*   **`document_chunks` Table:** Stores raw text chunks for traditional RAG.
*   **In-Database Logic:** Advanced graph algorithms (Shortest Path, PageRank-style importance) are implemented directly in SQL/PLpgSQL for efficiency.

## Prerequisites

*   **PostgreSQL 14+**
*   **Extensions:** `vector` (pgvector), `pg_trgm`, `uuid-ossp`.
*   **Python 3.x**
*   **Google Cloud API Key** (for Gemini embeddings).

## Setup & Installation

### 1. Database Setup

First, create the database and apply the schema which enables extensions and creates the necessary tables and functions.

```bash
# Switch to postgres user (if necessary)
sudo su - postgres

# Create database
createdb knowledge_graph_agent

# Apply schema
psql -d knowledge_graph_agent -f schema.sql
```

### 2. Environment Configuration

Create a `.env` file in this directory with your credentials:

```ini
GOOGLE_API_KEY=your_google_api_key
DB_USER=postgres
DB_PASSWORD=your_password
DB_HOST=localhost
DB_PORT=5432
DB_NAME=knowledge_graph_agent
```

### 3. Install Dependencies

Install the required Python packages:

```bash
pip install google-generativeai psycopg python-dotenv
# OR if using uv
uv pip install google-generativeai psycopg python-dotenv
```

## Usage

### Loading Sample Data

Use the provided script to generate embeddings and populate the graph with initial data. This script demonstrates how to insert nodes and edges while automatically generating vector embeddings for the node content.

```bash
python load_sample_data.py
# OR
uv run load_sample_data.py
```

### Agent Integration (Langflow)

The file `Agentic Hybrid-Graph RAG 2.0.json` is a flow definition for **Langflow** (or compatible agent orchestration tools). It defines an agent with access to two custom tools:

1.  **`HybridVectorTool` (Semantic Search):** Uses Reciprocal Rank Fusion (RRF) to combine vector similarity search with keyword matching.
2.  **`AdvancedGraphTool` (Graph Exploration):** Performs multi-hop traversals, shortest path finding, and relationship analysis.

**To use the agent:**
1.  Import the JSON file into Langflow.
2.  Ensure your LM Studio or other LLM provider is running (default configuration uses LM Studio).
3.  The agent is prompted to intelligently select between "semantic search" (for conceptual queries) and "graph exploration" (for structural/relationship queries).
