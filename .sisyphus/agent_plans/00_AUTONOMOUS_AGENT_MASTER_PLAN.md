# AUTONOMOUS 24/7 RESEARCH AGENT - COMPREHENSIVE PLAN

**Version**: 1.0
**Date**: January 13, 2026
**Status**: Planning Phase
**Priority**: P0 (Foundation)

---

## EXECUTIVE SUMMARY

**Objective**: Build an autonomous research agent that continuously gathers, ingests, and processes knowledge 24/7, automatically populating the knowledge_base PostgreSQL database.

**Key Deliverables**:
- 24/7 autonomous research agent daemon
- Automatic knowledge ingestion pipeline
- PostgreSQL integration for direct knowledge storage
- WSL-compatible file system access
- LLM-powered extraction with frontier models
- Continuous monitoring and error recovery

**Impact**: Foundation for all future improvements (GNN, KG completion, etc.)

---

## ARCHITECTURE OVERVIEW

### System Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                  AUTONOMOUS RESEARCH AGENT                    │
│                                                               │
│  ┌──────────────────────────────────────────────────────────────┐ │
│  │              ORCHESTRATION LAYER                         │ │
│  │  - Task Queue Management                                 │ │
│  │  - Scheduler (24/7)                                    │ │
│  │  - Resource Allocation                                    │ │
│  │  - Error Recovery                                        │ │
│  └────────────────────┬─────────────────────────────────────────┘ │
│                       │                                        │
│        ┌──────────────┼──────────────┬─────────────┐          │
│        ▼              ▼              ▼             ▼          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │Research  │  │ Ingestion │  │Processing │  │Storage   │   │
│  │  Agent   │  │  Pipeline │  │  Pipeline │  │  Layer   │   │
│  │          │  │           │  │           │  │          │   │
│  │• Search  │  │• LLM     │  │• Entity   │  │• Postgres │   │
│  │• Monitor │  │• Extract  │  │• Resolve  │  │• Direct   │   │
│  │• Queue   │  │• Glean   │  │• Cluster  │  │• Access  │   │
│  │          │  │           │  │• Summarize│  │          │   │
│  └─────┬────┘  └─────┬─────┘  └─────┬─────┘  └─────┬─────┘   │
│        │               │               │                │          │
│        └───────────────┴───────────────┴────────────────┘          │
│                       │                                        │
│                       ▼                                        │
│              ┌─────────────────┐                                │
│              │  EXTERNAL APIs │                                │
│              │                │                                │
│              │• LLM API       │                                │
│              │  (localhost:8317/v1)                             │
│              │                │                                │
│              │• PostgreSQL     │                                │
│              │  (localhost:5432)                                 │
│              │                │                                │
│              │• Web Search   │                                │
│              │  (Future)      │                                │
│              └─────────────────┘                                │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow

```
Research Queue
    │
    ▼
[URL/Source] → Research Agent
                  │
                  ▼
              [Download Content]
                  │
                  ▼
              [Text Content]
                  │
                  ▼
          [Ingestion Pipeline]
          ┌─────────────────────┐
          │ 1. Extract (LLM)   │
          │ 2. Resolve (GNN)   │
          │ 3. Store (Postgres)  │
          └─────────────────────┘
                  │
                  ▼
          [Processing Pipeline]
          ┌─────────────────────┐
          │ 4. Cluster        │
          │ 5. Summarize      │
          │ 6. Index         │
          └─────────────────────┘
                  │
                  ▼
          [Knowledge Graph in PostgreSQL]
```

---

## COMPONENT ARCHITECTURE

### 1. ORCHESTRATION LAYER

**Purpose**: Manage 24/7 operation, task scheduling, and error recovery.

**Core Components**:

#### 1.1 Task Queue System
```python
class TaskQueue:
    """Persistent task queue for 24/7 operation"""

    def __init__(self, db_conn: str):
        self.db_conn = db_conn

    async def add_task(self, task: ResearchTask):
        """Add new research task to queue"""
        pass

    async def get_next_task(self, worker_id: str) -> ResearchTask:
        """Get next pending task"""
        pass

    async def update_task_status(self, task_id: str, status: str):
        """Update task status (PENDING → IN_PROGRESS → COMPLETED/FAILED)"""
        pass

    async def get_failed_tasks(self, retry_limit: int = 3) -> List[ResearchTask]:
        """Get tasks that failed but are retryable"""
        pass
```

#### 1.2 Scheduler
```python
import asyncio
from apscheduler.schedulers.asyncio import AsyncIOScheduler

class AgentScheduler:
    """24/7 scheduler for research agent"""

    def __init__(self, task_queue: TaskQueue):
        self.scheduler = AsyncIOScheduler()
        self.task_queue = task_queue
        self.worker_pool = 5  # Concurrent workers

    async def start(self):
        """Start 24/7 operation"""
        # Add periodic tasks
        self.scheduler.add_job(
            self._process_queue,
            'interval',
            seconds=10,
            id='process_queue'
        )

        self.scheduler.add_job(
            self._processing_pipeline,
            'interval',
            seconds=60,
            id='process_pipeline'
        )

        self.scheduler.add_job(
            self._monitoring,
            'interval',
            seconds=300,  # 5 minutes
            id='monitoring'
        )

        self.scheduler.start()

    async def _process_queue(self):
        """Process research tasks"""
        pass

    async def _processing_pipeline(self):
        """Run processing pipeline on ingested data"""
        pass

    async def _monitoring(self):
        """Health checks and metrics"""
        pass
```

#### 1.3 Error Recovery
```python
class ErrorRecovery:
    """Automatic error recovery and retry logic"""

    def __init__(self, max_retries: int = 3, backoff_factor: float = 2.0):
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor

    async def execute_with_retry(self, func: Callable, *args, **kwargs):
        """Execute function with exponential backoff"""

        for attempt in range(self.max_retries):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise  # Final attempt failed

                backoff = self.backoff_factor ** attempt
                logger.error(f"Attempt {attempt+1} failed: {e}. Retrying in {backoff}s")
                await asyncio.sleep(backoff)

        raise MaxRetriesExceeded()
```

---

### 2. RESEARCH AGENT LAYER

**Purpose**: Discover, download, and queue content for ingestion.

**Core Components**:

#### 2.1 Source Discovery
```python
class SourceDiscovery:
    """Discover research sources (URLs, RSS feeds, APIs)"""

    def __init__(self, sources_config: str = "research_sources.yaml"):
        self.sources = self._load_sources(sources_config)

    def _load_sources(self, config_path: str) -> List[Dict]:
        """Load configured research sources"""
        pass

    async def discover_new_content(self) -> List[ResearchSource]:
        """Check for new content from configured sources"""
        pass

    async def add_manual_source(self, url: str, metadata: Dict):
        """Add manual research source"""
        pass
```

#### 2.2 Content Fetcher
```python
import aiohttp
import asyncio

class ContentFetcher:
    """Async content fetcher with rate limiting"""

    def __init__(self, max_concurrent: int = 10, rate_limit: float = 2.0):
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.rate_limit = rate_limit  # requests per second
        self.last_request_time = 0.0

    async def fetch_url(self, url: str) -> str:
        """Fetch content from URL with rate limiting"""

        async with self.semaphore:
            # Rate limiting
            await self._rate_limit_wait()

            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        return await response.text()
                    else:
                        raise FetchError(f"HTTP {response.status}")

    async def _rate_limit_wait(self):
        """Enforce rate limiting"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time

        if time_since_last < (1.0 / self.rate_limit):
            wait_time = (1.0 / self.rate_limit) - time_since_last
            await asyncio.sleep(wait_time)

        self.last_request_time = time.time()

    async def fetch_batch(self, urls: List[str]) -> Dict[str, str]:
        """Fetch multiple URLs concurrently"""
        tasks = [self.fetch_url(url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        return {url: result for url, result in zip(urls, results) if not isinstance(result, Exception)}
```

#### 2.3 Content Extractor
```python
from bs4 import BeautifulSoup
import re

class ContentExtractor:
    """Extract clean text content from various sources"""

    def extract_text(self, content: str, content_type: str) -> str:
        """Extract text content based on content type"""

        if content_type == 'html':
            return self._extract_from_html(content)
        elif content_type == 'markdown':
            return self._extract_from_markdown(content)
        elif content_type == 'pdf':
            return self._extract_from_pdf(content)
        else:
            return self._extract_from_plain_text(content)

    def _extract_from_html(self, html: str) -> str:
        """Extract clean text from HTML"""
        soup = BeautifulSoup(html, 'html.parser')

        # Remove script/style elements
        for element in soup(['script', 'style', 'nav', 'footer', 'header']):
            element.decompose()

        # Extract text
        text = soup.get_text(separator='\n')

        # Clean up
        text = re.sub(r'\n{3,}', '\n\n', text)  # Remove excessive newlines
        text = text.strip()

        return text

    def _extract_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF"""
        import PyPDF2
        with open(pdf_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            text = ''
            for page in reader.pages:
                text += page.extract_text()
        return text
```

---

### 3. INGESTION PIPELINE LAYER

**Purpose**: Extract structured knowledge from unstructured text using LLMs.

**Core Components**:

#### 3.1 LLM Ingestor (Wrapper around existing)
```python
from knowledge_base.ingestor import GraphIngestor

class AsyncIngestor(GraphIngestor):
    """Async wrapper around existing GraphIngestor"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.semaphore = asyncio.Semaphore(3)  # Limit concurrent LLM calls

    async def extract_async(self, text: str) -> KnowledgeGraph:
        """Async extraction with concurrency control"""

        async with self.semaphore:
            # Run sync extraction in thread pool
            loop = asyncio.get_event_loop()
            graph = await loop.run_in_executor(
                None,
                lambda: asyncio.run(super().extract(text))
            )
            return graph
```

#### 3.2 Direct PostgreSQL Storage
```python
import psycopg
from psycopg import sql

class DirectPostgresStorage:
    """Direct PostgreSQL access for knowledge storage"""

    def __init__(self, conn_str: str = "postgresql://agentzero@localhost:5432/knowledge_graph"):
        self.conn_str = conn_str
        self.pool = None

    async def initialize(self):
        """Initialize connection pool"""
        self.pool = await psycopg.AsyncConnectionPool(
            self.conn_str,
            min_size=5,
            max_size=20
        )

    async def store_entity(
        self,
        name: str,
        entity_type: str,
        description: str,
        embedding: List[float]
    ) -> str:
        """Store entity directly, return UUID"""

        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    INSERT INTO nodes (name, type, description, embedding)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (name, type)
                    DO UPDATE SET description = EXCLUDED.description,
                                    embedding = EXCLUDED.embedding,
                                    updated_at = NOW()
                    RETURNING id
                    """,
                    (name, entity_type, description, embedding)
                )
                row = await cur.fetchone()
                return str(row[0])

    async def store_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: str,
        description: str,
        weight: float = 1.0
    ) -> bool:
        """Store edge directly"""

        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    INSERT INTO edges (source_id, target_id, type, description, weight)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (source_id, target_id, type) DO NOTHING
                    """,
                    (source_id, target_id, edge_type, description, weight)
                )
                await conn.commit()
                return True

    async def store_event(
        self,
        node_id: str,
        description: str,
        timestamp: str,
        raw_time: str
    ) -> bool:
        """Store event directly"""

        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    INSERT INTO events (node_id, description, timestamp, raw_time_desc)
                    VALUES (%s, %s, %s, %s)
                    """,
                    (node_id, description, timestamp, raw_time)
                )
                await conn.commit()
                return True
```

#### 3.3 Ingestion Pipeline
```python
class IngestionPipeline:
    """Full ingestion pipeline coordination"""

    def __init__(
        self,
        ingestor: AsyncIngestor,
        storage: DirectPostgresStorage,
        embedder: EmbeddingGenerator
    ):
        self.ingestor = ingestor
        self.storage = storage
        self.embedder = embedder

    async def ingest_content(
        self,
        content: str,
        metadata: Dict = None
    ) -> Dict:
        """Full ingestion pipeline"""

        logger.info(f"Starting ingestion: {metadata.get('source', 'unknown')}")

        # 1. Extract knowledge graph
        graph = await self.ingestor.extract_async(content)

        # 2. Store entities with embeddings
        entity_id_map = {}
        for entity in graph.entities:
            embedding = await self.embedder.generate_embedding(
                f"{entity.name} {entity.description}"
            )
            entity_id = await self.storage.store_entity(
                entity.name,
                entity.type,
                entity.description,
                embedding
            )
            entity_id_map[entity.name] = entity_id

        # 3. Store edges
        for edge in graph.relationships:
            source_id = entity_id_map.get(edge.source)
            target_id = entity_id_map.get(edge.target)

            if source_id and target_id:
                await self.storage.store_edge(
                    source_id,
                    target_id,
                    edge.type,
                    edge.description,
                    edge.weight
                )

        # 4. Store events
        for event in graph.events:
            node_id = entity_id_map.get(event.primary_entity)
            if node_id:
                timestamp = event.normalized_date or None
                await self.storage.store_event(
                    node_id,
                    event.description,
                    timestamp,
                    event.raw_time
                )

        return {
            'status': 'completed',
            'entities_stored': len(graph.entities),
            'edges_stored': len(graph.relationships),
            'events_stored': len(graph.events),
            'source': metadata.get('source') if metadata else None
        }
```

---

### 4. PROCESSING PIPELINE LAYER

**Purpose**: Post-processing after ingestion (clustering, summarization, indexing).

**Core Components**:

#### 4.1 Trigger Management
```python
class ProcessingTrigger:
    """Determine when to run processing pipeline"""

    def __init__(self, min_entities: int = 100, min_time_hours: int = 1):
        self.min_entities = min_entities
        self.min_time_hours = min_time_hours
        self.last_processing_time = None

    async def should_process(self) -> bool:
        """Check if processing should run"""

        # Check entity count
        entity_count = await self._get_entity_count()
        if entity_count < self.min_entities:
            return False

        # Check time since last processing
        if self.last_processing_time:
            time_since = (datetime.now() - self.last_processing_time).total_seconds()
            if time_since < (self.min_time_hours * 3600):
                return False

        return True

    async def _get_entity_count(self) -> int:
        """Get current entity count from DB"""
        pass
```

#### 4.2 Processing Coordinator
```python
from knowledge_base.community import CommunityDetector
from knowledge_base.summarizer import CommunitySummarizer

class ProcessingCoordinator:
    """Coordinate community detection and summarization"""

    def __init__(self, storage: DirectPostgresStorage):
        self.storage = storage
        self.community_detector = None
        self.summarizer = None

    async def initialize(self):
        """Initialize components"""
        self.community_detector = CommunityDetector(self.storage.conn_str)
        self.summarizer = CommunitySummarizer(self.storage.conn_str)

    async def run_processing_pipeline(self) -> Dict:
        """Run full processing pipeline"""

        logger.info("Starting processing pipeline")

        # 1. Load graph
        G = await self.community_detector.load_graph()

        if G.number_of_nodes() == 0:
            return {'status': 'skipped', 'reason': 'empty_graph'}

        # 2. Detect communities
        logger.info(f"Detecting communities for {G.number_of_nodes()} nodes")
        memberships = self.community_detector.detect_communities(G)

        # 3. Save communities
        await self.community_detector.save_communities(memberships)

        # 4. Summarize communities
        logger.info("Summarizing communities")
        await self.summarizer.summarize_all()

        return {
            'status': 'completed',
            'nodes_processed': G.number_of_nodes(),
            'communities_created': len(set(m['cluster_id'] for m in memberships)),
            'summarized': True
        }
```

---

## MONITORING & LOGGING

### 1. Metrics Collection

```python
from prometheus_client import Counter, Histogram, Gauge

# Metrics
ingestion_total = Counter('agent_ingestion_total', 'Total ingestion operations')
ingestion_duration = Histogram('agent_ingestion_duration_seconds', 'Ingestion duration')
entities_stored = Gauge('agent_entities_stored', 'Total entities in DB')
processing_duration = Histogram('agent_processing_duration_seconds', 'Processing duration')
errors_total = Counter('agent_errors_total', 'Total errors', ['error_type'])

class MetricsCollector:
    """Collect and expose metrics"""

    def __init__(self):
        pass

    def record_ingestion(self, source: str, duration: float, entities: int):
        ingestion_total.labels(source=source).inc()
        ingestion_duration.observe(duration)
        entities_stored.set(entities)

    def record_processing(self, duration: float):
        processing_duration.observe(duration)

    def record_error(self, error_type: str):
        errors_total.labels(error_type=error_type).inc()
```

### 2. Health Checks

```python
class HealthChecker:
    """Health checks for all components"""

    async def check_health(self) -> Dict[str, bool]:
        """Check health of all components"""

        health = {}

        # Database connection
        health['database'] = await self._check_database()

        # LLM API
        health['llm_api'] = await self._check_llm_api()

        # Task queue
        health['task_queue'] = await self._check_task_queue()

        # Disk space
        health['disk_space'] = self._check_disk_space()

        return health

    async def _check_database(self) -> bool:
        """Check database connectivity"""
        try:
            async with self.storage.pool.connection() as conn:
                await conn.execute("SELECT 1")
            return True
        except Exception:
            return False

    async def _check_llm_api(self) -> bool:
        """Check LLM API availability"""
        try:
            models = await self.ingestor.list_available_models()
            return len(models) > 0
        except Exception:
            return False
```

---

## CONFIGURATION

### Configuration File (research_agent_config.yaml)

```yaml
agent:
  name: "autonomous_research_agent"
  version: "1.0"
  log_level: "INFO"

database:
  connection_string: "postgresql://agentzero@localhost:5432/knowledge_graph"
  pool_min_size: 5
  pool_max_size: 20

llm:
  base_url: "http://localhost:8317/v1"
  api_key: "lm-studio"
  model_default: "gemini-2.5-flash"
  model_pro: "gemini-2.5-pro"
  max_retries: 3
  timeout: 120  # seconds

embedding:
  provider: "google"
  model: "models/text-embedding-004"
  api_key_env: "GOOGLE_API_KEY"
  dimensions: 768

research:
  sources_config: "research_sources.yaml"
  max_concurrent_fetches: 10
  rate_limit: 2.0  # requests per second
  max_content_length: 1000000  # characters

ingestion:
  max_concurrent_llm_calls: 3
  batch_size: 10
  retry_backoff_factor: 2.0
  max_retries: 3

processing:
  min_entities_to_process: 100
  min_time_between_processing_hours: 1
  processing_interval_seconds: 60

monitoring:
  metrics_port: 8000
  health_check_interval_seconds: 300
  log_retention_days: 30

paths:
  knowledge_base: "\\\\wsl.localhost\\Ubuntu\\home\\administrator\\dev\\unbiased-observer\\knowledge_base"
  cache_dir: "./cache"
  logs_dir: "./logs"
  state_dir: "./state"
```

### Research Sources Configuration (research_sources.yaml)

```yaml
sources:
  - type: "manual"
    name: "Manual Input"
    description: "Manually added research sources"

  # Future: Add RSS feeds, APIs, etc.
  - type: "rss"
    name: "ArXiv CS.AI"
    url: "http://export.arxiv.org/rss/cs.AI.xml"
    update_interval_hours: 6

  - type: "api"
    name: "News API"
    url: "https://api.example.com/news"
    api_key_env: "NEWS_API_KEY"
    update_interval_hours: 1
```

---

## DEPLOYMENT ARCHITECTURE

### Directory Structure

```
unbiased-observer/
├── knowledge_base/                    # Existing KB code
│   ├── ingestor.py
│   ├── resolver.py
│   ├── community.py
│   ├── summarizer.py
│   └── ...
│
├── research_agent/                    # NEW: Autonomous agent
│   ├── __init__.py
│   ├── main.py                       # Entry point
│   ├── config.py                     # Configuration
│   ├── orchestrator/
│   │   ├── __init__.py
│   │   ├── scheduler.py
│   │   ├── task_queue.py
│   │   └── error_recovery.py
│   ├── research/
│   │   ├── __init__.py
│   │   ├── source_discovery.py
│   │   ├── content_fetcher.py
│   │   └── content_extractor.py
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── pipeline.py
│   │   ├── async_ingestor.py
│   │   └── postgres_storage.py
│   ├── processing/
│   │   ├── __init__.py
│   │   ├── coordinator.py
│   │   └── trigger.py
│   └── monitoring/
│       ├── __init__.py
│       ├── metrics.py
│       └── health_checker.py
│
├── configs/                            # Configuration files
│   ├── research_agent_config.yaml
│   └── research_sources.yaml
│
├── logs/                               # Logs
│   ├── agent.log
│   ├── ingestion.log
│   └── processing.log
│
└── state/                              # Agent state
    └── agent_state.json
```

### System Services

```yaml
# systemd service: /etc/systemd/system/autonomous-research-agent.service
[Unit]
Description=Autonomous 24/7 Research Agent
After=network.target postgresql.service

[Service]
Type=simple
User=administrator
WorkingDirectory=\\\\wsl.localhost\\Ubuntu\\home\\administrator\\dev\\unbiased-observer
Environment="PATH=/home/administrator/.local/bin:/usr/local/bin:/usr/bin:/bin"
ExecStart=/home/administrator/.local/bin/uv run research_agent/main.py
Restart=always
RestartSec=10
StandardOutput=append:/home/administrator/dev/unbiased-observer/logs/agent.log
StandardError=append:/home/administrator/dev/unbiased-observer/logs/agent_error.log

[Install]
WantedBy=multi-user.target
```

---

## SUCCESS CRITERIA

### 24/7 Operation
- [x] Agent runs continuously without manual intervention
- [ ] Automatic recovery from errors (retries, backoff)
- [ ] Graceful shutdown (completes in-flight tasks)
- [ ] No task loss (persistent queue)

### Knowledge Ingestion
- [ ] Automatic content fetching from configured sources
- [ ] LLM-based extraction working correctly
- [ ] Direct PostgreSQL storage operational
- [ ] >90% ingestion success rate

### Processing Pipeline
- [ ] Automatic community detection trigger
- [ ] Automatic summarization trigger
- [ ] Processing completes within reasonable time (<1 hour for 1K entities)

### Monitoring & Observability
- [ ] Prometheus metrics exposed
- [ ] Health check endpoint available
- [ ] Comprehensive logging (all operations)
- [ ] Error tracking and alerting

### Data Quality
- [ ] Entity resolution quality maintained (F1 > 0.85)
- [ ] No duplicate entities (unique constraint enforced)
- [ ] Temporal data accurately stored
- [ ] Community structure maintained

---

## ROLLBACK STRATEGY

### Feature Flags
```python
# config.py
FEATURE_FLAGS = {
    'auto_ingestion': True,
    'auto_processing': True,
    'use_gnn_embeddings': False,  # Future
    'enable_kg_completion': False  # Future
}
```

### Rollback Triggers
- Ingestion failure rate >20% for 1 hour
- Processing pipeline hangs (>4 hours)
- Database errors >10% of operations
- Memory usage >90%

### Rollback Actions
1. Disable auto-ingestion
2. Pause processing pipeline
3. Preserve queue state (don't lose tasks)
4. Investigate logs
5. Re-enable after fix

---

## SECURITY CONSIDERATIONS

### 1. Input Validation
```python
def validate_content(content: str, max_length: int = 1000000) -> bool:
    """Validate content before ingestion"""

    # Length check
    if len(content) > max_length:
        raise ValueError(f"Content too long: {len(content)} > {max_length}")

    # Malicious pattern detection
    import re
    dangerous_patterns = [
        r'<script\b',
        r'javascript:',
        r'on\w+\s*=',
    ]

    for pattern in dangerous_patterns:
        if re.search(pattern, content, re.IGNORECASE):
            raise ValueError(f"Potentially malicious content detected: {pattern}")

    return True
```

### 2. SQL Injection Prevention
- Use parameterized queries (already done in storage layer)
- Never concatenate SQL strings
- Validate all inputs

### 3. Rate Limiting
- Implement per-source rate limits
- Global rate limiting for LLM API calls
- Backoff on 429 errors

---

## TIMELINE

### Phase 1: Foundation (Week 1)
- Day 1-2: Project structure, configuration
- Day 3-4: Task queue implementation
- Day 5-7: Scheduler and error recovery

### Phase 2: Research Agent (Week 2)
- Day 1-3: Content fetcher
- Day 4-5: Content extractor
- Day 6-7: Source discovery (basic manual)

### Phase 3: Ingestion Pipeline (Week 3)
- Day 1-3: Async ingestor wrapper
- Day 4-5: Direct PostgreSQL storage
- Day 6-7: Full ingestion pipeline

### Phase 4: Processing Pipeline (Week 4)
- Day 1-3: Processing coordinator
- Day 4-5: Trigger management
- Day 6-7: Integration with existing community/summarizer

### Phase 5: Monitoring & Deployment (Week 5)
- Day 1-3: Metrics collection
- Day 4-5: Health checks
- Day 6-7: Deployment (systemd, testing)

---

**Total Timeline**: 5 weeks to production-ready autonomous agent

---

**Document Status**: ✅ COMPLETE - READY FOR BREAKDOWN
**Next Step**: Break down into implementation tasks
