# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview
This repository implements an **Agentic Hybrid-Graph RAG** system ("Unbiased Observer"). It combines structured knowledge graphs with semantic vector search in PostgreSQL to enable autonomous agents to reason over complex relationships.

- **Language**: Python 3.10+
- **Database**: PostgreSQL 14+ with `vector`, `pg_trgm`, `uuid-ossp` extensions
- **Package Manager**: `uv`
- **Architecture**: Hybrid Storage (Nodes/Edges + Vectors) with In-Database Logic (PL/pgSQL)

## Development Environment
- **Environment Variables**: Ensure `.env` contains:
  - Database: `DB_USER`, `DB_PASSWORD`, `DB_HOST`, `DB_PORT`, `DB_NAME`
  - AI Services: `GOOGLE_API_KEY` (embeddings), `OPENAI_API_BASE` (LLM endpoint)
- **PYTHONPATH**: When running scripts directly, ensure `knowledge_base` is in your python path (e.g., `export PYTHONPATH=$PYTHONPATH:$(pwd)/knowledge_base`).

## Common Commands

### Testing
- **Run Master Test Suite**:
  ```bash
  ./run_tests.sh
  # OR
  export PYTHONPATH=$PYTHONPATH:$(pwd)/knowledge_base
  uv run knowledge_base/tests/master_test.py
  ```

### Operation
- **Ingest Data** (Run pipeline on a text file):
  ```bash
  uv run knowledge_base/pipeline.py path/to/your/data.txt
  ```
- **Visualize Hierarchy** (View knowledge tree):
  ```bash
  uv run knowledge_base/visualize.py
  ```

## Code Architecture

### Core Components (`knowledge_base/`)
- **`pipeline.py`**: Main entry point/orchestrator. Handles the end-to-end flow of ingesting documents.
- **`ingestor.py`**: Extraction engine using `instructor` + `openai`. Uses a 2-pass "gleaning" strategy for high-resolution extraction.
- **`resolver.py`**: Entity resolution logic. Combines vector similarity with LLM-based cognitive reasoning to deduplicate nodes.
- **`community.py`**: Implements Leiden algorithm for hierarchical clustering (Micro-communities to Macro-themes).
- **`summarizer.py`**: Recursive reporting engine that generates "Intelligence Reports" for communities.
- **`visualize.py`**: CLI tool for inspecting the knowledge hierarchy.
- **`langflow_tool.py`**: Integration component for Langflow.

### Database (`schema.sql`)
- Uses PostgreSQL with `pgvector`.
- Key tables: `nodes`, `edges`, `communities`, `community_membership`, `community_hierarchy`, `events`.

### Agent Orchestration (`rag/`)
- Contains Langflow definitions (`Agentic Hybrid-Graph RAG 2.0.json`) defining the agentic flow.

## Core Directives & Agent Protocols

### Role: Adaptive Project Architect
You are not just an executor; you are the **Project Architect**. Your goal is to maximize user efficiency by adapting your behavior and workflows.
- **Evolution Loop**: Observe user preferences -> Orient against rules -> Decide on updates -> Act.
- **Self-Correction**: If the user corrects you, apologize, fix the issue, and **IMMEDIATELY** propose an update to this file (`CLAUDE.md`) to prevent recurrence.

### Metaprotocols (Mental Sandbox)
Before acting on complex requests, engage these cognitive phases:
1. **`<thinking_journal>`**: Deconstruct the request, identify objectives/constraints, and log assumptions.
2. **`<recursive_self_improvement>`**: Evaluate your plan against the **Evaluation Criteria** (Accuracy, Efficiency, Conventions, Clarity, Coherence, Safety, Completeness) and refine it *before* execution.

### Operational Protocols
- **Planning (`PLANS.md`)**: This file is the source of truth for the roadmap and active tasks. Check it at the start of a session and update it after significant changes.
- **Context (`PROJECT_CONTEXT.md`)**: This is the "Brain". Read it to understand domain knowledge and architecture.
- **File Integrity**: Never truncate existing sections in markdown files.
- **Resilience**: Propose git commits after stable steps.

## Detailed Code Style Guidelines

### Python (3.10+)
- **Type Hints**: Mandatory for all function args/returns. Use modern syntax (`list[str]`, `str | None`).
- **Imports**: Absolute imports preferred. Sort: Stdlib > Third-party > Local.
- **Async/Await**: Use `async def` for I/O. Use `async with` for DB connections (`psycopg.AsyncConnection`).
- **Error Handling**: Log with `exc_info=True`. Chain exceptions (`raise ServiceError from e`).
- **Data Models**: Use Pydantic `BaseModel` with `Field` descriptions.
- **Formatting**: Use `ruff` for linting/formatting. Use f-strings exclusively.

### Naming Conventions
- Variables/Functions: `snake_case`
- Classes: `PascalCase`
- Constants: `UPPER_CASE`
- Private: Leading underscore (`_variable`)
