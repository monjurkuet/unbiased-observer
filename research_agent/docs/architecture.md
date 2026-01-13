# System Architecture

Detailed architecture overview of the Autonomous Research Agent system.

---

## Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
  - [High-Level Architecture](#high-level-architecture)
  - [Component Architecture](#component-architecture)
- [Data Flow](#data-flow)
  - [Task Lifecycle](#task-lifecycle)
  - [Content Processing Pipeline](#content-processing-pipeline)
  - [Knowledge Graph Construction](#knowledge-graph-construction)
- [Component Details](#component-details)
  - [Orchestrator](#orchestrator)
  - [Research Module](#research-module)
  - [Ingestion Pipeline](#ingestion-pipeline)
  - [Processing Coordinator](#processing-coordinator)
  - [Monitoring System](#monitoring-system)
  - [Web Interface](#web-interface)
- [Database Schema](#database-schema)
  - [Knowledge Graph Tables](#knowledge-graph-tables)
  - [Agent Operations Tables](#agent-operations-tables)
- [API Architecture](#api-architecture)
- [Deployment Architecture](#deployment-architecture)
- [Performance Considerations](#performance-considerations)

---

## Overview

The Autonomous Research Agent is a distributed, asynchronous system that continuously discovers, processes, and organizes research content into a structured knowledge graph. The system is designed for 24/7 operation with robust error recovery, scalable processing, and comprehensive monitoring.

### Key Architectural Principles

- **Asynchronous Processing**: All I/O operations use async/await patterns
- **Task-Based Architecture**: Work is divided into FETCH, INGEST, and PROCESS tasks
- **Modular Design**: Components are loosely coupled and independently testable
- **Persistent State**: All state is stored in PostgreSQL for reliability
- **Event-Driven**: Components communicate through database state changes
- **Scalable Storage**: Vector embeddings enable semantic search at scale

---

## System Architecture

### High-Level Architecture

```
┌──────────────────────────────────────────────────────┐
│         24/7 Agent Scheduler (APScheduler)          │
│     (Task Queue, Error Recovery, Health Monitoring) │
└──────────┬───────────────────────────────────────┘
           │
           ├─> FETCH Tasks ──> ContentFetcher
           │                      └─> ContentExtractor
           │
           ├─> INGEST Tasks ──> IngestionPipeline
           │                        ├─> AsyncIngestor
           │                        └─> DirectPostgresStorage
           │
           └─> PROCESS Tasks ──> ProcessingCoordinator
                                   ├─> CommunityDetector
                                   └─> CommunitySummarizer
```

### Component Architecture

```
Autonomous Research Agent
├── orchestrator/          # Task scheduling and queue management
│   ├── scheduler.py       # APScheduler-based 24/7 operation
│   ├── task_queue.py      # Persistent PostgreSQL task queue
│   └── error_recovery.py  # Exponential backoff retry logic
├── research/              # Content discovery and fetching
│   ├── arxiv_integrator.py    # arXiv API integration
│   ├── content_fetcher.py      # URL/file content fetching
│   ├── content_extractor.py    # Text extraction utilities
│   ├── source_discovery.py    # Automated source discovery
│   └── manual_source.py       # Manual source management
├── ingestion/             # Entity extraction and storage
│   ├── pipeline.py            # Main ingestion coordinator
│   ├── async_ingestor.py       # Async LLM-powered extraction
│   └── postgres_storage.py     # PostgreSQL storage layer
├── processing/            # Community detection and analysis
│   ├── coordinator.py         # Processing coordinator
│   └── trigger.py              # Processing triggers
├── monitoring/             # Health checks and metrics
│   ├── metrics.py             # Metrics collection
│   └── health_checker.py       # System health monitoring
├── ui/                    # Web interface
│   ├── app.py                  # Streamlit application
│   └── run.py                  # UI runner script
├── config.py              # Configuration management
├── main.py                # Main entry point
└── db_schema.sql          # Database schema
```

---

## Data Flow

### Task Lifecycle

The system operates through a three-phase task lifecycle:

#### Phase 1: FETCH (Content Discovery)

```
Source Discovery → Task Creation → Content Fetching → Content Extraction
```

1. **Source Discovery**: arXiv monitoring or manual source addition
2. **Task Creation**: FETCH tasks added to PostgreSQL task queue
3. **Content Fetching**: Asynchronous URL/file retrieval with rate limiting
4. **Content Extraction**: Text extraction from HTML, PDF, Markdown, plain text

#### Phase 2: INGEST (Entity Extraction)

```
Extracted Content → LLM Processing → Entity Storage → Relationship Storage → Event Storage
```

1. **LLM Processing**: Two-pass extraction (core + gleaning) using Google GenAI
2. **Entity Storage**: Batch insertion of nodes with vector embeddings
3. **Relationship Storage**: Directed edges between entities with weights
4. **Event Storage**: Temporal data linked to primary entities

#### Phase 3: PROCESS (Community Analysis)

```
Graph Loading → Community Detection → Community Storage → Summarization
```

1. **Graph Loading**: Full knowledge graph loaded from PostgreSQL
2. **Community Detection**: Leiden algorithm for clustering related entities
3. **Community Storage**: Membership assignments and hierarchy saved
4. **Summarization**: LLM-powered community descriptions generated

### Content Processing Pipeline

```
Raw Content
    ↓
ContentExtractor (HTML/PDF/Markdown/Text)
    ↓
IngestionPipeline
    ↓
AsyncIngestor (LLM Entity Extraction)
    ↓
DirectPostgresStorage (Batch Operations)
    ↓
Knowledge Graph (Nodes, Edges, Events)
```

### Knowledge Graph Construction

The knowledge graph is built through structured entity extraction:

#### Entity Types
- **Person**: Researchers, authors, contributors
- **Organization**: Universities, companies, research institutions
- **Concept**: Research topics, methodologies, technologies
- **Publication**: Papers, articles, books
- **Project**: Research projects, software, datasets

#### Relationship Types
- **authored_by**: Publication → Person
- **affiliated_with**: Person → Organization
- **cites**: Publication → Publication
- **uses_method**: Publication → Concept
- **related_to**: Concept → Concept

#### Temporal Events
- **Publication dates**: When research was published
- **Conference dates**: When work was presented
- **Timeline events**: Career milestones, project timelines

---

## Component Details

### Orchestrator

The orchestrator manages the 24/7 operation and task coordination:

#### AgentScheduler
- **APScheduler Integration**: Uses AsyncIOScheduler for non-blocking periodic jobs
- **Job Types**:
  - `process_queue()`: Task queue processing (10s intervals)
  - `process_ingestion_queue()`: Ingestion pipeline (immediate)
  - `retry_failed_tasks()`: Failed task recovery (5min intervals)
  - `monitoring()`: Health checks (5min intervals)
  - `arxiv_monitoring()`: Research discovery (2h intervals)
- **Concurrency Control**: max_instances=1 prevents job conflicts

#### TaskQueue
- **PostgreSQL Backend**: Persistent storage for task lifecycle
- **Task States**: PENDING → IN_PROGRESS → COMPLETED/FAILED
- **Concurrency Safety**: FOR UPDATE SKIP LOCKED prevents duplicate processing
- **Retry Logic**: Automatic retry_count incrementing with max_retries limit

#### ErrorRecovery
- **Exponential Backoff**: Configurable retry delays (base_delay * 2^retry_count)
- **Decorator Pattern**: `@with_retry` for automatic error handling
- **Failure Thresholds**: Tasks marked permanently FAILED after max_retries

### Research Module

Handles content discovery and initial processing:

#### ArxivIntegrator
- **API Wrapper**: arXiv API client with search and metadata retrieval
- **Search Methods**:
  - `search_by_keywords()`: Keyword-based paper discovery
  - `search_by_category()`: Category-based filtering (cs.AI, cs.LG, etc.)
  - `get_paper_details()`: Full paper metadata and abstract
- **Automated Monitoring**: `run_monitoring_cycle()` for continuous discovery

#### ContentFetcher
- **Asynchronous Fetching**: aiohttp-based URL retrieval with rate limiting
- **Content Types**: URLs, local files, direct text input
- **Error Handling**: Retry logic with exponential backoff
- **Rate Limiting**: Semaphore-based concurrency control

#### ContentExtractor
- **Format Detection**: Automatic content type identification
- **Text Extraction**:
  - HTML: Script/style removal, content cleaning
  - Markdown: Formatting preservation
  - PDF: Text extraction (requires additional dependencies)
  - Plain Text: Direct processing
- **Length Limits**: Configurable max_content_length truncation

### Ingestion Pipeline

Transforms raw content into structured knowledge:

#### IngestionPipeline
- **Coordinator**: Orchestrates the 4-stage ingestion process
- **Stages**:
  1. Extract entities from content
  2. Store entities in database
  3. Store relationships between entities
  4. Store temporal events
- **Performance Tracking**: Timing and metrics collection

#### AsyncIngestor
- **LLM Integration**: Google GenAI for entity extraction
- **Two-Pass Extraction**:
  - **Core Pass**: Primary entity and relationship identification
  - **Gleaning Pass**: Secondary relationship discovery
- **Concurrency Control**: Semaphore-based limiting (max 3 concurrent)
- **Structured Output**: Pydantic models for type safety

#### DirectPostgresStorage
- **Batch Operations**: Efficient bulk inserts with conflict resolution
- **Vector Embeddings**: Automatic embedding generation for semantic search
- **Connection Pooling**: Async connection management
- **Transaction Safety**: Atomic operations with rollback on failure

### Processing Coordinator

Analyzes the accumulated knowledge graph:

#### ProcessingCoordinator
- **Pipeline Management**: 4-stage processing workflow
- **Integration**: Loads graph → detects communities → saves results → summarizes
- **Performance Metrics**: Processing time, entity counts, community statistics

#### ProcessingTrigger
- **Smart Scheduling**: Prevents unnecessary processing
- **Thresholds**:
  - `min_entities_to_process`: Minimum entities before processing
  - `min_time_between_processing_hours`: Cooldown period
- **State Tracking**: Last processing timestamp in agent_state table

### Monitoring System

Provides operational visibility:

#### MetricsCollector
- **Event Tracking**: Task lifecycles, ingestion operations, processing jobs
- **Structured Logging**: Separate log files for different components
- **Performance Data**: Duration tracking, success/failure rates
- **Summary Reports**: Real-time statistics and health status

#### HealthChecker
- **System Checks**: Database connectivity, LLM API availability
- **Automated Monitoring**: Periodic health assessments
- **Alert Integration**: Ready for external monitoring systems

### Web Interface

User-facing visualization and interaction:

#### Streamlit Application
- **Dashboard**: Real-time statistics and system status
- **Knowledge Graph**: Interactive network visualization with Plotly
- **Query Interface**: Natural language research queries
- **Analytics**: Charts for research trends and distributions

#### UI Architecture
- **Mock Data**: Currently uses simulated data (ready for API integration)
- **Responsive Design**: Works on desktop and mobile
- **Real-time Updates**: WebSocket-ready for live data

---

## Database Schema

### Knowledge Graph Tables

#### nodes
```sql
CREATE TABLE nodes (
    id SERIAL PRIMARY KEY,
    name VARCHAR(500) NOT NULL,
    type VARCHAR(100) NOT NULL,
    description TEXT,
    embedding vector(768),  -- Vector embedding for semantic search
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

#### edges
```sql
CREATE TABLE edges (
    id SERIAL PRIMARY KEY,
    source_id INTEGER REFERENCES nodes(id),
    target_id INTEGER REFERENCES nodes(id),
    type VARCHAR(100) NOT NULL,
    weight FLOAT DEFAULT 1.0,
    description TEXT,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(source_id, target_id, type)
);
```

#### events
```sql
CREATE TABLE events (
    id SERIAL PRIMARY KEY,
    primary_entity_id INTEGER REFERENCES nodes(id),
    event_type VARCHAR(100) NOT NULL,
    event_date DATE,
    description TEXT,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

#### communities
```sql
CREATE TABLE communities (
    id SERIAL PRIMARY KEY,
    name VARCHAR(500),
    description TEXT,
    summary TEXT,
    size INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

#### community_membership
```sql
CREATE TABLE community_membership (
    community_id INTEGER REFERENCES communities(id),
    node_id INTEGER REFERENCES nodes(id),
    importance FLOAT,
    PRIMARY KEY (community_id, node_id)
);
```

### Agent Operations Tables

#### research_tasks
```sql
CREATE TABLE research_tasks (
    id SERIAL PRIMARY KEY,
    task_type VARCHAR(50) NOT NULL,  -- FETCH, INGEST, PROCESS
    status VARCHAR(50) DEFAULT 'PENDING',
    priority VARCHAR(20) DEFAULT 'medium',
    payload JSONB,
    result JSONB,
    error_message TEXT,
    retry_count INTEGER DEFAULT 0,
    max_retries INTEGER DEFAULT 3,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP,
    completed_at TIMESTAMP
);
```

#### research_sources
```sql
CREATE TABLE research_sources (
    id SERIAL PRIMARY KEY,
    type VARCHAR(50) NOT NULL,
    name VARCHAR(200) NOT NULL,
    config JSONB,
    active BOOLEAN DEFAULT true,
    priority VARCHAR(20) DEFAULT 'medium',
    last_fetched TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

#### ingestion_logs
```sql
CREATE TABLE ingestion_logs (
    id SERIAL PRIMARY KEY,
    task_id INTEGER REFERENCES research_tasks(id),
    content_length INTEGER,
    entities_extracted INTEGER,
    relationships_extracted INTEGER,
    processing_time_seconds FLOAT,
    success BOOLEAN,
    error_message TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

#### processing_logs
```sql
CREATE TABLE processing_logs (
    id SERIAL PRIMARY KEY,
    task_id INTEGER REFERENCES research_tasks(id),
    nodes_processed INTEGER,
    communities_created INTEGER,
    processing_time_seconds FLOAT,
    success BOOLEAN,
    error_message TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

#### agent_state
```sql
CREATE TABLE agent_state (
    key VARCHAR(100) PRIMARY KEY,
    value JSONB,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

---

## API Architecture

### Internal APIs

#### Task Queue API
- **add_task()**: Queue new tasks with priority and payload
- **get_next_task()**: Retrieve next available task with locking
- **update_task_status()**: Update task progress and results
- **increment_retry()**: Handle task retry logic

#### Storage API
- **store_entity()**: Store knowledge graph nodes with embeddings
- **store_entities_batch()**: Bulk entity insertion
- **store_edge()**: Store relationships between entities
- **store_edges_batch()**: Bulk relationship insertion
- **store_event()**: Store temporal events

#### Research API
- **search_by_keywords()**: arXiv keyword search
- **search_by_category()**: arXiv category filtering
- **fetch_url()**: Asynchronous content retrieval
- **extract_text()**: Content format normalization

### External APIs

#### LLM API (Google GenAI)
- **generate_content()**: Entity extraction and summarization
- **embed_content()**: Vector embedding generation
- **batch_embed()**: Bulk embedding operations

#### Database API (PostgreSQL)
- **Connection Pooling**: Async connection management
- **Vector Operations**: pgvector similarity search
- **Transaction Management**: ACID compliance for data integrity

---

## Deployment Architecture

### Development Deployment

```
Local Machine
├── Python 3.10+
├── PostgreSQL 14+
├── LLM API (Local/Cloud)
└── File System Storage
```

### Production Deployment

```
Production Server
├── Systemd Service (research-agent.service)
├── PostgreSQL Database
├── LLM API Access
├── Log Rotation
├── Monitoring Integration
└── Backup System
```

### Cloud Deployment

```
Cloud Infrastructure
├── Container Orchestration (Docker/Kubernetes)
├── Managed PostgreSQL (RDS/Aurora)
├── API Gateway (for LLM access)
├── Load Balancer
├── Auto-scaling Groups
└── Monitoring Stack (Prometheus/Grafana)
```

---

## Performance Considerations

### Scalability Factors

#### Database Performance
- **Connection Pooling**: 5-20 connections based on load
- **Vector Indexing**: pgvector HNSW indexes for fast similarity search
- **Partitioning**: Time-based partitioning for large datasets
- **Query Optimization**: Composite indexes on frequently queried columns

#### Processing Performance
- **Batch Operations**: Bulk inserts/updates for efficiency
- **Concurrency Limits**: Semaphore-based control of concurrent operations
- **Memory Management**: Streaming for large content processing
- **Caching**: In-memory caches for frequently accessed data

#### Network Performance
- **Rate Limiting**: Respectful API usage patterns
- **Connection Reuse**: Persistent connections for external APIs
- **Timeout Management**: Configurable timeouts with retry logic
- **Compression**: Content compression for large payloads

### Monitoring Metrics

#### System Metrics
- **Task Throughput**: Tasks processed per minute/hour
- **Queue Depth**: Pending tasks in queue
- **Error Rates**: Failed task percentages
- **Processing Times**: Average task completion times

#### Performance Metrics
- **Database Latency**: Query execution times
- **API Response Times**: External service latencies
- **Memory Usage**: RAM consumption patterns
- **CPU Utilization**: Processing load distribution

#### Business Metrics
- **Content Discovery**: New papers/sources found per day
- **Knowledge Growth**: Entities added to graph per day
- **Community Formation**: New research communities detected
- **Query Performance**: Knowledge graph query response times

---

## Security Architecture

### Data Protection
- **API Key Management**: Environment variables, never in code
- **Database Credentials**: Encrypted configuration files
- **Log Security**: Sensitive data redaction in logs
- **Access Control**: Configurable user permissions

### Operational Security
- **Rate Limiting**: Respectful API usage patterns
- **Error Handling**: No sensitive information in error messages
- **Audit Trails**: Complete logging of all operations
- **Backup Strategy**: Automated database backups

### Network Security
- **HTTPS Only**: All external API calls use secure protocols
- **Certificate Validation**: SSL/TLS verification enabled
- **Firewall Rules**: Restrict database and service access
- **VPN Access**: Secure remote management

---

## Future Architecture Evolution

### Phase 6-10 Roadmap

#### Multi-Agent Architecture (Phase 8)
- **Agent Orchestration**: Multiple specialized agents
- **Inter-Agent Communication**: Message passing protocols
- **Distributed Processing**: Cross-agent task coordination

#### Enterprise Integration (Phase 10)
- **RESTful APIs**: External system integration
- **Plugin Architecture**: Extensible component system
- **Multi-Tenancy**: Isolated knowledge graphs per organization
- **Advanced Security**: Enterprise-grade authentication and authorization

#### Advanced Analytics (Phase 6)
- **Citation Networks**: Academic citation analysis
- **Trend Detection**: Research velocity and impact metrics
- **Collaboration Mapping**: Author and institution networks
- **Predictive Modeling**: Research trend forecasting

---

## Troubleshooting Architecture Issues

### Common Performance Bottlenecks

1. **Database Connection Pool Exhaustion**
   - **Symptom**: Connection timeouts, slow queries
   - **Solution**: Increase pool_max_size, optimize queries

2. **LLM API Rate Limiting**
   - **Symptom**: API quota exceeded errors
   - **Solution**: Implement exponential backoff, reduce concurrency

3. **Memory Issues**
   - **Symptom**: Out of memory errors during processing
   - **Solution**: Reduce batch sizes, implement streaming

4. **Task Queue Backlog**
   - **Symptom**: Growing pending task count
   - **Solution**: Increase worker concurrency, optimize processing

### Monitoring and Alerting

- **Health Checks**: Database connectivity, API availability
- **Performance Thresholds**: Response time alerts, error rate monitoring
- **Capacity Planning**: Resource utilization tracking
- **Automated Recovery**: Self-healing mechanisms for common issues

---

**Last Updated**: January 14, 2026
