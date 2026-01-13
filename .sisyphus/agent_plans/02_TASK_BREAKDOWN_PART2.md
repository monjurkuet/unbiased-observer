# AUTONOMOUS AGENT - TASK BREAKDOWN PART 2

**Version**: 1.0
**Date**: January 13, 2026
**Status**: Ready for Implementation

---

## PHASE 2: RESEARCH AGENT (WEEK 2) - CONTINUED

### Task 2.3: Source Discovery
**Priority**: P1 | **Effort**: 3 hours | **Dependencies**: 1.2

**File**: `research_agent/research/source_discovery.py`

**Implementation**:
```python
import yaml
from pathlib import Path
import logging
from typing import List, Dict

logger = logging.getLogger('research_agent.research')

class SourceDiscovery:
    """Discover and manage research sources"""

    def __init__(self, sources_config: str):
        self.sources_config = sources_config
        self.sources = self._load_sources(sources_config)

    def _load_sources(self, config_path: str) -> List[Dict]:
        """Load configured research sources from YAML"""

        config_file = Path(config_path)
        if not config_file.exists():
            logger.warning(f"Config file not found: {config_path}")
            return [{'type': 'manual', 'name': 'Manual Input', 'active': True}]

        with open(config_file, 'r') as f:
            config_data = yaml.safe_load(f)

        sources = config_data.get('sources', [])
        logger.info(f"Loaded {len(sources)} research sources")

        return sources

    async def discover_new_content(self) -> List[Dict]:
        """Check for new content from configured sources"""

        logger.info("Discovering new content from sources...")

        discovered = []

        for source in self.sources:
            if not source.get('active', True):
                logger.debug(f"Skipping inactive source: {source.get('name')}")
                continue

            if source['type'] == 'manual':
                logger.debug("Manual source - no auto-discovery")
                continue

            # Future: Add RSS, API discovery logic
            logger.warning(f"Source type {source['type']} not yet implemented")

        return discovered

    async def add_manual_source(self, url: str, metadata: Dict = None) -> str:
        """Add manual research source to database"""

        import psycopg

        source_id = str(uuid.uuid4())

        # Store in database (implementation in ingestion phase)
        logger.info(f"Added manual source: {source_id}")

        return source_id

    def get_sources(self) -> List[Dict]:
        """Get all configured sources"""
        return self.sources

    def get_active_sources(self) -> List[Dict]:
        """Get only active sources"""
        return [s for s in self.sources if s.get('active', True)]
```

**Success Criteria**:
- [ ] Sources load from YAML
- [ ] Manual sources can be added
- [ ] Inactive sources skipped
- [ ] Discovery ready for RSS/API (future)

**Verification**: Load test config, verify sources parsed correctly

---

### Task 2.4: Manual Source Interface
**Priority**: P1 | **Effort**: 2 hours | **Dependencies**: 2.3

**File**: `research_agent/research/manual_source.py` (new file)

**Implementation**:
```python
from research_agent.research.source_discovery import SourceDiscovery
from research_agent.orchestrator.task_queue import TaskQueue
import logging

logger = logging.getLogger('research_agent.research')

class ManualSourceManager:
    """Interface for manually adding research sources"""

    def __init__(self, task_queue: TaskQueue, discovery: SourceDiscovery):
        self.task_queue = task_queue
        self.discovery = discovery

    async def add_url_source(self, url: str, metadata: dict = None) -> str:
        """Add URL as research source and create task"""

        logger.info(f"Adding URL source: {url}")

        # Validate URL
        if not self._validate_url(url):
            raise ValueError(f"Invalid URL: {url}")

        # Create source record
        source_id = await self.discovery.add_manual_source(url, metadata)

        # Create FETCH task
        task_id = await self.task_queue.add_task(
            task_type='FETCH',
            source=url,
            metadata={
                'source_id': source_id,
                'source_type': 'url',
                'added_by': 'manual',
                **(metadata or {})
            }
        )

        logger.info(f"Created task {task_id} for URL {url}")
        return task_id

    async def add_file_source(self, file_path: str, metadata: dict = None) -> str:
        """Add file as research source and create task"""

        logger.info(f"Adding file source: {file_path}")

        # Validate file path
        if not Path(file_path).exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Support WSL paths
        wsl_path = file_path.replace('\\\\', '\\')

        # Create source record
        source_id = await self.discovery.add_manual_source(
            file_path,
            metadata or {}
        )

        # Create FETCH task
        task_id = await self.task_queue.add_task(
            task_type='FETCH',
            source=file_path,
            metadata={
                'source_id': source_id,
                'source_type': 'file',
                'wsl_path': wsl_path,
                'added_by': 'manual',
                **(metadata or {})
            }
        )

        logger.info(f"Created task {task_id} for file {file_path}")
        return task_id

    async def add_text_source(self, text: str, metadata: dict = None) -> str:
        """Add text directly as research source"""

        logger.info(f"Adding text source ({len(text)} chars)")

        # Create temporary file or store directly
        source_id = await self.discovery.add_manual_source(
            f"text_source_{len(text)}",
            metadata or {}
        )

        # Create INGEST task directly (skip fetch)
        task_id = await self.task_queue.add_task(
            task_type='INGEST',
            source='direct_text',
            metadata={
                'source_id': source_id,
                'source_type': 'text',
                'text_content': text,
                'added_by': 'manual',
                **(metadata or {})
            }
        )

        logger.info(f"Created task {task_id} for direct text")
        return task_id

    def _validate_url(self, url: str) -> bool:
        """Validate URL format"""

        import re
        url_pattern = re.compile(
            r'^https?://'  # http or https
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+'  # subdomain
            r'(?:[A-Z]{2,6}\.?)'  # top-level domain
            r'[A-Z]{2,6}'  # top level domain (e.g., com)
            r'(?::\d{1,5})?)'  # optional port
            r'(?:/?|[/?]\S+)?$', re.IGNORECASE
        )

        return bool(url_pattern.match(url))
```

**Success Criteria**:
- [ ] URL sources can be added
- [ ] File sources can be added (WSL compatible)
- [ ] Text sources can be added directly
- [ ] URL validation works

**Verification**: Test adding URL, file, and text sources

---

### Task 2.5: Async Ingestor Wrapper
**Priority**: P0 | **Effort**: 4 hours | **Dependencies**: 1.2, 1.4

**File**: `research_agent/ingestion/async_ingestor.py`

**Implementation**:
```python
import sys
import asyncio
from pathlib import Path
import logging
from knowledge_base.ingestor import GraphIngestor, KnowledgeGraph

logger = logging.getLogger('research_agent.ingestion')

class AsyncIngestor(GraphIngestor):
    """Async wrapper around existing GraphIngestor"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Concurrency control for LLM calls
        self.semaphore = asyncio.Semaphore(3)

    async def extract_async(self, text: str) -> KnowledgeGraph:
        """Async extraction with concurrency control"""

        logger.info(f"Starting async extraction ({len(text)} chars)")

        async with self.semaphore:
            # Run sync extraction in thread pool
            loop = asyncio.get_event_loop()
            graph = await loop.run_in_executor(
                None,
                lambda: super().extract(text)
            )

        logger.info(
            f"Extraction complete: "
            f"{len(graph.entities)} entities, "
            f"{len(graph.relationships)} relationships, "
            f"{len(graph.events)} events"
        )

        return graph

    async def batch_extract_async(
        self,
        texts: list[str],
        batch_size: int = 10
    ) -> list[KnowledgeGraph]:
        """Batch extract multiple texts"""

        logger.info(f"Batch extracting {len(texts)} texts (batch_size={batch_size})")

        results = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")

            # Extract batch concurrently
            tasks = [self.extract_async(text) for text in batch]
            batch_results = await asyncio.gather(*tasks)

            results.extend(batch_results)

        logger.info(f"Batch extraction complete: {len(results)} graphs")
        return results
```

**Success Criteria**:
- [ ] Async extraction works
- [ ] Semaphore limits concurrent LLM calls
- [ ] Batch processing works
- [ ] Thread pool execution works

**Verification**: Test extracting multiple texts concurrently

---

## PHASE 3: INGESTION PIPELINE (WEEK 3) - 3 TASKS

### Task 3.1: Direct PostgreSQL Storage Layer
**Priority**: P0 | **Effort**: 5 hours | **Dependencies**: 1.3, 1.2

**File**: `research_agent/ingestion/postgres_storage.py`

**Implementation**:
```python
import psycopg
from psycopg import sql
from typing import List
import logging
from google.generativeai import genai
import os

logger = logging.getLogger('research_agent.ingestion')

class DirectPostgresStorage:
    """Direct PostgreSQL access for knowledge storage"""

    def __init__(self, config):
        self.conn_str = config.database.connection_string
        self.pool = None
        self.embedding_api_key = os.getenv('GOOGLE_API_KEY')

    async def initialize(self):
        """Initialize connection pool"""

        if self.pool is None:
            self.pool = await psycopg.AsyncConnectionPool(
                self.conn_str,
                min_size=5,
                max_size=20
            )
            logger.info("PostgreSQL storage initialized")

    async def close(self):
        """Close connection pool"""

        if self.pool:
            await self.pool.close()
            logger.info("PostgreSQL storage closed")

    async def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding using Google GenAI"""

        if not self.embedding_api_key:
            logger.warning("GOOGLE_API_KEY missing, using dummy embedding")
            return [0.0] * 768

        try:
            genai.configure(api_key=self.embedding_api_key)
            result = genai.embed_content(
                model="models/text-embedding-004",
                content=text,
                task_type="retrieval_document"
            )
            return result['embedding']
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}", exc_info=True)
            # Fallback to dummy embedding
            return [0.0] * 768

    async def store_entity(
        self,
        name: str,
        entity_type: str,
        description: str,
        embedding: List[float] = None
    ) -> str:
        """Store entity directly, return UUID"""

        # Generate embedding if not provided
        if embedding is None:
            embedding = await self.generate_embedding(f"{name} {description}")

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
                if row is None:
                    # Conflict occurred, fetch existing
                    await cur.execute(
                        "SELECT id FROM nodes WHERE name = %s AND type = %s",
                        (name, entity_type)
                    )
                    row = await cur.fetchone()

        entity_id = str(row[0])
        logger.debug(f"Stored entity: {entity_id} ({name})")
        return entity_id

    async def store_entities_batch(
        self,
        entities: List[dict]
    ) -> dict:
        """Store multiple entities in batch"""

        logger.info(f"Storing {len(entities)} entities in batch")

        entity_id_map = {}

        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                # Use execute_batch for efficiency
                query = sql.SQL("""
                    INSERT INTO nodes (name, type, description, embedding)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (name, type)
                    DO UPDATE SET description = EXCLUDED.description,
                                    embedding = EXCLUDED.embedding,
                                    updated_at = NOW()
                    RETURNING id
                """)

                # Prepare data with embeddings
                data = []
                for entity in entities:
                    embedding = entity.get('embedding')
                    if embedding is None:
                        embedding = await self.generate_embedding(
                            f"{entity['name']} {entity['description']}"
                        )
                    data.append((
                        entity['name'],
                        entity['type'],
                        entity['description'],
                        embedding
                    ))

                # Execute batch
                await cur.executemany(query, data)

                # Fetch returned IDs
                for i, (row,) in enumerate(await cur):
                    entity_id = str(row[0])
                    entity_id_map[entities[i]['name']] = entity_id

        logger.info(f"Batch stored {len(entity_id_map)} entities")
        return entity_id_map

    async def store_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: str,
        description: str = None,
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

        logger.debug(f"Stored edge: {source_id} --[{edge_type}]--> {target_id}")
        return True

    async def store_edges_batch(self, edges: List[dict], entity_id_map: dict) -> int:
        """Store multiple edges in batch"""

        logger.info(f"Storing {len(edges)} edges in batch")

        stored = 0

        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                query = sql.SQL("""
                    INSERT INTO edges (source_id, target_id, type, description, weight)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (source_id, target_id, type) DO NOTHING
                """)

                # Prepare data
                data = []
                for edge in edges:
                    source_id = entity_id_map.get(edge['source'])
                    target_id = entity_id_map.get(edge['target'])

                    if source_id and target_id:
                        data.append((
                            source_id,
                            target_id,
                            edge['type'],
                            edge.get('description'),
                            edge.get('weight', 1.0)
                        ))

                # Execute batch
                if data:
                    await cur.executemany(query, data)
                    stored = len(data)

        logger.info(f"Batch stored {stored} edges")
        return stored

    async def store_event(
        self,
        node_id: str,
        description: str,
        timestamp: str = None,
        raw_time: str = None
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

        logger.debug(f"Stored event: {node_id}")
        return True

    async def store_events_batch(self, events: List[dict], entity_id_map: dict) -> int:
        """Store multiple events in batch"""

        logger.info(f"Storing {len(events)} events in batch")

        stored = 0

        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                query = sql.SQL("""
                    INSERT INTO events (node_id, description, timestamp, raw_time_desc)
                    VALUES (%s, %s, %s, %s)
                """)

                # Prepare data
                data = []
                for event in events:
                    node_id = entity_id_map.get(event['primary_entity'])

                    if node_id:
                        data.append((
                            node_id,
                            event['description'],
                            event.get('normalized_date'),
                            event.get('raw_time')
                        ))

                # Execute batch
                if data:
                    await cur.executemany(query, data)
                    stored = len(data)

        logger.info(f"Batch stored {stored} events")
        return stored
```

**Success Criteria**:
- [ ] Entity storage works with embeddings
- [ ] Batch entity storage works
- [ ] Edge storage works
- [ ] Event storage works
- [ ] WSL path handling works

**Verification**: Write integration test for storage layer

---

### Task 3.2: Full Ingestion Pipeline Coordinator
**Priority**: P0 | **Effort**: 4 hours | **Dependencies**: 2.5, 3.1

**File**: `research_agent/ingestion/pipeline.py`

**Implementation**:
```python
from research_agent.ingestion.async_ingestor import AsyncIngestor
from research_agent.ingestion.postgres_storage import DirectPostgresStorage
from knowledge_base.ingestor import KnowledgeGraph
import logging
from typing import Dict

logger = logging.getLogger('research_agent.ingestion')

class IngestionPipeline:
    """Full ingestion pipeline coordination"""

    def __init__(self, config):
        self.config = config
        self.ingestor = AsyncIngestor(
            base_url=config.llm.base_url,
            api_key=config.llm.api_key,
            model_name=config.llm.model_default
        )
        self.storage = DirectPostgresStorage(config)

    async def initialize(self):
        """Initialize components"""

        await self.storage.initialize()
        logger.info("Ingestion pipeline initialized")

    async def ingest_content(
        self,
        content: str,
        metadata: Dict = None
    ) -> Dict:
        """Full ingestion pipeline from text content"""

        source = metadata.get('source', 'unknown') if metadata else 'unknown'
        logger.info(f"Starting ingestion: {source}")

        start_time = asyncio.get_event_loop().time()

        try:
            # 1. Extract knowledge graph
            logger.info("Stage 1: Extracting knowledge graph...")
            graph = await self.ingestor.extract_async(content)

            # 2. Store entities with embeddings
            logger.info("Stage 2: Storing entities...")
            entity_data = [
                {
                    'name': e.name,
                    'type': e.type,
                    'description': e.description
                }
                for e in graph.entities
            ]

            entity_id_map = await self.storage.store_entities_batch(entity_data)

            # 3. Store edges
            logger.info("Stage 3: Storing edges...")
            edge_data = [
                {
                    'source': r.source,
                    'target': r.target,
                    'type': r.type,
                    'description': r.description,
                    'weight': r.weight
                }
                for r in graph.relationships
            ]

            edges_stored = await self.storage.store_edges_batch(edge_data, entity_id_map)

            # 4. Store events
            logger.info("Stage 4: Storing events...")
            event_data = [
                {
                    'primary_entity': e.primary_entity,
                    'description': e.description,
                    'normalized_date': e.normalized_date,
                    'raw_time': e.raw_time
                }
                for e in graph.events
            ]

            events_stored = await self.storage.store_events_batch(event_data, entity_id_map)

            # Calculate duration
            end_time = asyncio.get_event_loop().time()
            duration = end_time - start_time

            result = {
                'status': 'completed',
                'entities_stored': len(graph.entities),
                'edges_stored': edges_stored,
                'events_stored': events_stored,
                'duration_seconds': duration,
                'source': source
            }

            logger.info(
                f"Ingestion complete: "
                f"{len(graph.entities)} entities, "
                f"{edges_stored} edges, "
                f"{events_stored} events, "
                f"duration: {duration:.2f}s"
            )

            return result

        except Exception as e:
            logger.error(f"Ingestion failed: {e}", exc_info=True)
            return {
                'status': 'failed',
                'error': str(e),
                'source': source
            }

    async def ingest_from_fetch_result(
        self,
        fetch_result: Dict
    ) -> Dict:
        """Ingest content from fetch result"""

        content = fetch_result.get('content')
        metadata = fetch_result.get('metadata', {})

        if not content:
            logger.warning("No content to ingest")
            return {'status': 'skipped', 'reason': 'no_content'}

        return await self.ingest_content(content, metadata)
```

**Success Criteria**:
- [ ] Full pipeline works end-to-end
- [ ] All 4 stages execute correctly
- [ ] Error handling works
- [ ] Duration tracked

**Verification**: Test ingestion with sample content

---

### Task 3.3: Ingestion Queue Handler
**Priority**: P0 | **Effort**: 3 hours | **Dependencies**: 1.7, 3.2

**File**: Update `research_agent/orchestrator/scheduler.py` with ingest handler

**Implementation**:
```python
# Add to AgentScheduler class:

async def _handle_ingest_task(self, task: 'ResearchTask'):
    """Handle ingestion task"""

    logger.info(f"Handling INGEST task: {task.id}")

    try:
        # Get content from task metadata
        source = task.source
        metadata = task.metadata

        content = None

        # Check if text content provided directly
        if metadata.get('source_type') == 'text':
            content = metadata.get('text_content')
            logger.info("Using direct text content from task")

        # Otherwise, load from file/source
        else:
            # Import fetcher here to re-use
            from research_agent.research.content_fetcher import ContentFetcher
            fetcher = ContentFetcher(self.config)

            if metadata.get('source_type') == 'file':
                # Read from WSL-compatible file path
                content = await fetcher.fetch_file(source)
            elif metadata.get('source_type') == 'url':
                # Fetch from URL
                content = await fetcher.fetch_url(source)

        if not content:
            await self.task_queue.update_task_status(
                task.id,
                'FAILED',
                error_message='No content to ingest'
            )
            return

        # Initialize pipeline
        from research_agent.ingestion.pipeline import IngestionPipeline
        pipeline = IngestionPipeline(self.config)
        await pipeline.initialize()

        # Run ingestion pipeline
        result = await pipeline.ingest_content(content, {
            'source': source,
            **metadata
        })

        # Update task status
        if result['status'] == 'completed':
            await self.task_queue.update_task_status(task.id, 'COMPLETED')

            # Log to ingestion_logs table
            await self._log_ingestion_result(task.id, result)

        else:
            await self.task_queue.update_task_status(
                task.id,
                'FAILED',
                error_message=result.get('error', 'Unknown error')
            )

    except Exception as e:
        logger.error(f"Error in ingest task: {e}", exc_info=True)
        await self.task_queue.increment_retry(task.id)

async def _log_ingestion_result(self, task_id: str, result: Dict):
    """Log ingestion result to database"""

    async with self.task_queue.pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                INSERT INTO ingestion_logs
                (task_id, status, entities_stored, edges_stored, events_stored, duration_seconds)
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (
                    task_id,
                    result['status'],
                    result.get('entities_stored', 0),
                    result.get('edges_stored', 0),
                    result.get('events_stored', 0),
                    result.get('duration_seconds', 0)
                )
            )
```

**Success Criteria**:
- [ ] INGEST tasks processed correctly
- [ ] Direct text content handled
- [ ] File content loaded (WSL compatible)
- [ ] Results logged to database

**Verification**: Test INGEST task with file and text source

---

## PHASE 4: PROCESSING PIPELINE (WEEK 4) - 2 TASKS

### Task 4.1: Processing Coordinator
**Priority**: P1 | **Effort**: 6 hours | **Dependencies**: 1.2, existing community/summarizer

**File**: `research_agent/processing/coordinator.py`

**Implementation**:
```python
import sys
from pathlib import Path
import logging
from knowledge_base.community import CommunityDetector
from knowledge_base.summarizer import CommunitySummarizer

logger = logging.getLogger('research_agent.processing')

class ProcessingCoordinator:
    """Coordinate community detection and summarization"""

    def __init__(self, config):
        self.config = config

        # Add knowledge_base to path
        kb_path = Path(config.paths.knowledge_base)
        sys.path.insert(0, str(kb_path))

        self.community_detector = None
        self.summarizer = None

    async def initialize(self):
        """Initialize components"""

        self.community_detector = CommunityDetector(self.config.database.connection_string)
        self.summarizer = CommunitySummarizer(
            self.config.database.connection_string,
            base_url=self.config.llm.base_url,
            api_key=self.config.llm.api_key,
            model_name=self.config.llm.model_pro
        )

        logger.info("Processing coordinator initialized")

    async def run_processing_pipeline(self) -> Dict:
        """Run full processing pipeline"""

        logger.info("Starting processing pipeline")

        start_time = asyncio.get_event_loop().time()

        try:
            # 1. Load graph
            logger.info("Stage 1: Loading graph...")
            G = await self.community_detector.load_graph()

            if G.number_of_nodes() == 0:
                return {'status': 'skipped', 'reason': 'empty_graph'}

            # 2. Detect communities
            logger.info("Stage 2: Detecting communities...")
            memberships = self.community_detector.detect_communities(G)

            # 3. Save communities
            logger.info("Stage 3: Saving communities...")
            await self.community_detector.save_communities(memberships)

            # 4. Summarize communities
            logger.info("Stage 4: Summarizing communities...")
            await self.summarizer.summarize_all()

            # Calculate metrics
            end_time = asyncio.get_event_loop().time()
            duration = end_time - start_time

            unique_communities = len(set(m['cluster_id'] for m in memberships))

            result = {
                'status': 'completed',
                'nodes_processed': G.number_of_nodes(),
                'communities_created': unique_communities,
                'duration_seconds': duration
            }

            logger.info(
                f"Processing complete: "
                f"{G.number_of_nodes()} nodes, "
                f"{unique_communities} communities, "
                f"duration: {duration:.2f}s"
            )

            return result

        except Exception as e:
            logger.error(f"Processing failed: {e}", exc_info=True)
            return {
                'status': 'failed',
                'error': str(e)
            }

    async def should_process(self) -> bool:
        """Check if processing should run"""

        # Check entity count
        from psycopg import AsyncConnection
        async with await AsyncConnection.connect(
            self.config.database.connection_string
        ) as conn:
            async with conn.cursor() as cur:
                await cur.execute("SELECT COUNT(*) FROM nodes")
                row = await cur.fetchone()
                entity_count = row[0]

        min_entities = self.config.processing.min_entities_to_process

        if entity_count < min_entities:
            logger.debug(f"Not enough entities to process: {entity_count} < {min_entities}")
            return False

        # Check time since last processing
        # (Implementation would need last_processing_time tracking)
        return True
```

**Success Criteria**:
- [ ] Community detection works
- [ ] Summarization works
- [ ] Full pipeline coordinates correctly
- [ ] Should-process logic works

**Verification**: Test processing pipeline on small graph

---

### Task 4.2: Processing Trigger Management
**Priority**: P1 | **Effort**: 3 hours | **Dependencies**: 4.1

**File**: `research_agent/processing/trigger.py`

**Implementation**:
```python
import logging

logger = logging.getLogger('research_agent.processing')

class ProcessingTrigger:
    """Determine when to run processing pipeline"""

    def __init__(self, config):
        self.config = config.processing

    async def should_trigger(self) -> bool:
        """Check if processing should trigger"""

        # 1. Check entity count
        entity_count_ok = await self._check_entity_count()

        if not entity_count_ok:
            logger.debug("Entity count threshold not met")
            return False

        # 2. Check time interval
        time_ok = await self._check_time_interval()

        if not time_ok:
            logger.debug("Time interval not met")
            return False

        return True

    async def _check_entity_count(self) -> bool:
        """Check if entity count meets minimum"""

        from psycopg import AsyncConnection
        async with await AsyncConnection.connect(
            self.config.database.connection_string
        ) as conn:
            async with conn.cursor() as cur:
                await cur.execute("SELECT COUNT(*) FROM nodes")
                row = await cur.fetchone()
                entity_count = row[0]

        min_entities = self.config.min_entities_to_process

        result = entity_count >= min_entities

        logger.info(f"Entity count check: {entity_count} >= {min_entities} = {result}")
        return result

    async def _check_time_interval(self) -> bool:
        """Check if minimum time has passed since last processing"""

        # Get last processing time from agent_state
        from psycopg import AsyncConnection
        async with await AsyncConnection.connect(
            self.config.database.connection_string
        ) as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    SELECT value FROM agent_state
                    WHERE key = 'last_processing_time'
                    """
                )
                row = await cur.fetchone()

                if not row:
                    # No previous processing, should run
                    logger.info("No previous processing time found")
                    return True

                last_time_str = row[0]
                # Parse JSON value
                if isinstance(last_time_str, str):
                    import json
                    last_time = json.loads(last_time_str)
                    from datetime import datetime
                    last_time = datetime.fromisoformat(last_time)
                else:
                    last_time = last_time_str

                from datetime import datetime, timedelta
                min_interval = timedelta(
                    hours=self.config.min_time_between_processing_hours
                )

                current_time = datetime.now()
                time_since = current_time - last_time

                result = time_since >= min_interval

                logger.info(
                    f"Time interval check: "
                    f"{time_since.total_seconds()}s >= {min_interval.total_seconds()}s = {result}"
                )

                return result

    async def record_processing_time(self):
        """Record processing time in agent_state"""

        from datetime import datetime
        import json

        current_time = datetime.now().isoformat()

        from psycopg import AsyncConnection
        async with await AsyncConnection.connect(
            self.config.database.connection_string
        ) as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    INSERT INTO agent_state (key, value)
                    VALUES ('last_processing_time', %s)
                    ON CONFLICT (key) DO UPDATE
                    SET value = EXCLUDED.value, updated_at = NOW()
                    """,
                    (current_time,)
                )

        logger.info(f"Recorded processing time: {current_time}")
```

**Success Criteria**:
- [ ] Entity count check works
- [ ] Time interval check works
- [ ] Should-trigger logic works
- [ ] Last processing time tracked

**Verification**: Test trigger conditions

---

## PHASE 5: MONITORING & DEPLOYMENT (WEEK 5) - 4 TASKS

### Task 5.1: Metrics Collection
**Priority**: P1 | **Effort**: 4 hours | **Dependencies**: None

**File**: `research_agent/monitoring/metrics.py`

**Implementation**:
```python
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import logging

logger = logging.getLogger('research_agent.monitoring')

# Metrics
ingestion_total = Counter(
    'agent_ingestion_total',
    'Total ingestion operations',
    ['source_type']
)
ingestion_duration = Histogram(
    'agent_ingestion_duration_seconds',
    'Ingestion duration in seconds',
    ['source_type']
)
entities_stored = Gauge(
    'agent_entities_stored',
    'Total entities in database'
)
edges_stored = Gauge(
    'agent_edges_stored',
    'Total edges in database'
)
processing_duration = Histogram(
    'agent_processing_duration_seconds',
    'Processing duration in seconds'
)
tasks_pending = Gauge(
    'agent_tasks_pending',
    'Number of pending tasks'
)
tasks_failed = Counter(
    'agent_tasks_failed_total',
    'Total failed tasks',
    ['task_type']
)

class MetricsCollector:
    """Collect and expose metrics"""

    def __init__(self, port: int = 8000):
        self.port = port

    def start_server(self):
        """Start Prometheus metrics server"""

        try:
            start_http_server(self.port)
            logger.info(f"Prometheus server started on port {self.port}")
        except Exception as e:
            logger.error(f"Failed to start metrics server: {e}", exc_info=True)

    def record_ingestion(
        self,
        source_type: str,
        duration: float,
        entities: int,
        edges: int,
        events: int
    ):
        """Record ingestion metrics"""

        ingestion_total.labels(source_type=source_type).inc()
        ingestion_duration.labels(source_type=source_type).observe(duration)
        entities_stored.set(entities)
        edges_stored.set(edges)

        logger.debug(
            f"Metrics: ingestion_duration={duration:.2f}s, "
            f"entities={entities}, edges={edges}"
        )

    def record_processing(self, duration: float, nodes: int, communities: int):
        """Record processing metrics"""

        processing_duration.observe(duration)

        logger.debug(
            f"Metrics: processing_duration={duration:.2f}s, "
            f"nodes={nodes}, communities={communities}"
        )

    def record_task_pending(self, count: int):
        """Record pending tasks count"""

        tasks_pending.set(count)

    def record_task_failed(self, task_type: str):
        """Record failed task"""

        tasks_failed.labels(task_type=task_type).inc()
```

**Success Criteria**:
- [ ] Prometheus server starts
- [ ] All metrics exposed
- [ ] Metrics can be scraped

**Verification**: Start server, curl metrics endpoint

---

### Task 5.2: Health Check System
**Priority**: P1 | **Effort**: 3 hours | **Dependencies**: 5.1

**File**: `research_agent/monitoring/health_checker.py`

**Implementation**:
```python
import logging
from typing import Dict
from research_agent.ingestion.postgres_storage import DirectPostgresStorage

logger = logging.getLogger('research_agent.monitoring')

class HealthChecker:
    """Health checks for all components"""

    def __init__(self, config):
        self.config = config
        self.storage = DirectPostgresStorage(config)

    async def check_health(self) -> Dict[str, Dict]:
        """Check health of all components"""

        health = {}

        # 1. Database connection
        health['database'] = await self._check_database()

        # 2. LLM API
        health['llm_api'] = await self._check_llm_api()

        # 3. Task queue
        health['task_queue'] = await self._check_task_queue()

        # 4. Scheduler
        health['scheduler'] = await self._check_scheduler()

        # Overall status
        all_healthy = all(v['healthy'] for v in health.values())
        health['overall'] = {
            'status': 'healthy' if all_healthy else 'degraded',
            'checks': health
        }

        return health

    async def _check_database(self) -> Dict:
        """Check database connectivity"""

        try:
            await self.storage.initialize()
            await self.storage.close()

            return {'healthy': True, 'message': 'Database OK'}
        except Exception as e:
            return {'healthy': False, 'message': f'Database error: {e}'}

    async def _check_llm_api(self) -> Dict:
        """Check LLM API availability"""

        try:
            from research_agent.ingestion.async_ingestor import AsyncIngestor
            ingestor = AsyncIngestor(
                base_url=self.config.llm.base_url,
                api_key=self.config.llm.api_key
            )

            models = await ingestor.list_available_models()

            is_healthy = len(models) > 0

            return {
                'healthy': is_healthy,
                'message': f'LLM API OK ({len(models)} models)' if is_healthy else 'LLM API unavailable'
            }
        except Exception as e:
            return {'healthy': False, 'message': f'LLM API error: {e}'}

    async def _check_task_queue(self) -> Dict:
        """Check task queue status"""

        try:
            from research_agent.orchestrator.task_queue import TaskQueue
            queue = TaskQueue(self.config.database.connection_string)
            await queue.initialize()

            pending_count = await queue.get_pending_count()

            is_healthy = pending_count < 1000  # Reasonable threshold

            return {
                'healthy': is_healthy,
                'message': f'Task queue OK ({pending_count} pending)' if is_healthy else f'Task queue backed up ({pending_count} pending)'
            }
        except Exception as e:
            return {'healthy': False, 'message': f'Task queue error: {e}'}

    async def _check_scheduler(self) -> Dict:
        """Check scheduler status"""

        # Placeholder - would check if scheduler is running
        return {
            'healthy': True,
            'message': 'Scheduler OK'
        }

    async def get_health_summary(self) -> str:
        """Get human-readable health summary"""

        health = await self.check_health()

        lines = []
        lines.append("=" * 50)
        lines.append("AUTONOMOUS RESEARCH AGENT - HEALTH STATUS")
        lines.append("=" * 50)
        lines.append(f"Overall Status: {health['overall']['status'].upper()}")
        lines.append("")

        for component, status in health['overall']['checks'].items():
            if component == 'overall':
                continue

            status_emoji = '✓' if status['healthy'] else '✗'
            lines.append(f"{status_emoji} {component.upper()}: {status['message']}")

        lines.append("=" * 50)

        return '\n'.join(lines)
```

**Success Criteria**:
- [ ] Database health check works
- [ ] LLM API health check works
- [ ] Task queue health check works
- [ ] Health summary readable

**Verification**: Run health checks, verify output

---

[CONTINUED IN NEXT DOCUMENT...]
```

**Status**: Tasks 1.1-4.2 detailed. Tasks 5.1-5.2 above. Remaining in next file.

**Next**: Create Phase 5-6 and configuration files breakdown
