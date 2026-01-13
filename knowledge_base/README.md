# Knowledge Base GraphRAG System

This directory contains the high-fidelity **Agentic GraphRAG** ingestion and management system. It extends standard RAG by adding structured relationships, hierarchical community summaries, and chronological timelines. The system provides both a REST API and web interface for knowledge graph exploration and management.

## Features
- **High-Resolution Extraction:** Uses a 2-pass "gleaning" strategy with Google Gemini models to capture subtle entities and relationships.
- **Hybrid Entity Resolution:** Combines vector similarity with LLM-based cognitive reasoning to deduplicate nodes (e.g., merging "Dr. Vance" and "Elena Vance").
- **Hierarchical Clustering:** Uses the **Leiden Algorithm** to cluster nodes into Micro-communities, which are then rolled up into Macro-themes.
- **Recursive Summarization:** Automatically generates "Intelligence Reports" for every community. Parent communities summarize their children, creating a searchable map of knowledge.
- **Temporal Tracking:** Extracts specific events and dates to build structured timelines.
- **REST API:** Complete FastAPI-based REST API for programmatic access.
- **Web Interface:** Modern Streamlit-based UI for interactive exploration.
- **Real-time Updates:** WebSocket support for progress tracking during long operations.
- **Production Ready:** Enterprise-grade configuration, logging, and deployment options.

## Architecture

The system consists of three main layers:

### 1. Core Processing Layer
- `pipeline.py`: Main orchestrator for document ingestion and processing
- `ingestor.py`: LLM-powered entity and relationship extraction
- `resolver.py`: Entity deduplication and resolution using embeddings
- `community.py`: Hierarchical community detection using Leiden algorithm
- `summarizer.py`: Recursive summarization of communities

### 2. API Layer
- `api.py`: FastAPI endpoints for all operations
- `websocket.py`: Real-time communication for long-running tasks
- `main_api.py`: API server entry point
- `config.py`: Configuration management system

### 3. User Interface Layer
- `streamlit-ui/app.py`: Web-based interface for exploration and management
- Interactive graph visualization, search, and content ingestion

### 4. Supporting Components
- `visualize.py`: CLI visualization tools
- `schema.sql`: Database schema and indexes
- `config.py`: Centralized configuration management

## Quick Start

### 1. Requirements
Ensure your `.env` file in the root directory has the following:
- `DB_USER`, `DB_PASSWORD`, `DB_HOST`, `DB_PORT`, `DB_NAME`
- `GOOGLE_API_KEY` (for embeddings and LLM operations)
- `LLM_MODEL` (default: gemini-2.5-flash)
- `API_HOST`, `API_PORT` (default: 0.0.0.0:8000)
- `STREAMLIT_API_URL`, `STREAMLIT_WS_URL` (for UI configuration)

### 2. Install Dependencies
```bash
pip install -r requirements.txt
# or
uv pip install -r requirements.txt
```

### 3. Ingest Data (CLI)
```bash
uv run python pipeline.py path/to/your/data.txt
```

### 4. Start the API Server
```bash
uv run python main_api.py
```
The API will be available at `http://localhost:8000` with documentation at `http://localhost:8000/docs`

### 5. Start the Web Interface
```bash
cd streamlit-ui
uv run streamlit run app.py
```
The web interface will be available at `http://localhost:8501`

### 6. Run Both Services
For full functionality, run both the API server and web interface:

**Terminal 1 - API Server:**
```bash
uv run python main_api.py
```

**Terminal 2 - Web Interface:**
```bash
cd streamlit-ui && uv run streamlit run app.py
```

### 7. Alternative: CLI Visualization
```bash
uv run python visualize.py
```

## API Endpoints

The system provides a REST API for programmatic access:

### Core Endpoints
- `GET /api/stats` - Get database statistics
- `GET /api/nodes` - Retrieve knowledge graph nodes
- `GET /api/edges` - Retrieve knowledge graph edges
- `GET /api/communities` - Get community information
- `GET /api/graph` - Get graph data for visualization
- `POST /api/search` - Search nodes by text query

### Ingestion Endpoints
- `POST /api/ingest/text` - Ingest text content directly
- `POST /api/ingest/file` - Upload and ingest text files

### Management Endpoints
- `POST /api/community/detect` - Run community detection
- `POST /api/summarize` - Run recursive summarization

### WebSocket Endpoints
- `WS /ws` - Real-time updates for operations
- `WS /ws/{channel}` - Channel-specific updates

## Web Interface

The Streamlit-based web interface provides:

### Dashboard
- Real-time database statistics
- Quick access to operations

### Search & Exploration
- Full-text search across knowledge graph
- Interactive graph visualization with Plotly
- Community analysis and exploration

### Content Ingestion
- Direct text input
- File upload support
- Real-time processing feedback

### Graph Visualization
- Interactive network graphs
- Node type filtering
- Community highlighting

## Documentation

- **[API Documentation](docs/API_DOCUMENTATION.md)** - Complete API reference with endpoints, request/response formats, and examples
- **[Deployment Guide](docs/DEPLOYMENT_GUIDE.md)** - Production deployment instructions, scaling, and monitoring
- **[Environment Configuration](.env.template)** - Template for environment variables
- **[Database Schema](schema.sql)** - PostgreSQL schema with pgvector extensions
- **[Design Document](docs/DESIGN.md)** - Architecture and implementation details
- **[Codebase Architecture](docs/CODEBASE_ARCHITECTURE.md)** - Complete codebase structure and component guide
- **[Production Readiness](docs/PRODUCTION_READINESS_SUMMARY.md)** - Production readiness assessment and improvements

## Database Schema
The system uses PostgreSQL with `pgvector`. See `schema.sql` for table definitions (`nodes`, `edges`, `communities`, `community_membership`, `community_hierarchy`, `events`).
