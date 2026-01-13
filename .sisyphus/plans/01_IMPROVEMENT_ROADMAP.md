# KNOWLEDGE BASE - IMPROVEMENT ROADMAP

**Version**: 1.0
**Date**: January 13, 2026
**Priority**: Strategic Planning Document

---

## EXECUTIVE SUMMARY

This document outlines a prioritized roadmap for improving the Knowledge Base system. Improvements are organized by impact, effort, and dependencies to guide development decisions.

**High Impact / Low Effort** (Quick Wins):
1. Add configuration management
2. Implement Redis caching
3. Add hybrid search (RRF fusion)
4. Add connection pooling

**High Impact / High Effort** (Strategic):
1. Graph Neural Networks for entity embeddings
2. Automated knowledge graph completion
3. Incremental updates
4. Multi-modal support

---

## 1. QUICK WINS (1-4 WEEKS)

### 1.1 Configuration Management (Effort: 2 days, Impact: High)

**Problem**: Model names, thresholds, and URLs scattered across code.

**Solution**: Create centralized configuration module.

**Implementation**:
```python
# config.py
from pydantic_settings import BaseSettings

class KBConfig(BaseSettings):
    # Database
    db_user: str
    db_password: str
    db_host: str = "localhost"
    db_port: int = 5432
    db_name: str

    # LLM Configuration
    llm_base_url: str = "http://localhost:8317/v1"
    llm_api_key: str = "lm-studio"
    llm_model_default: str = "gemini-2.5-flash"
    llm_model_pro: str = "gemini-2.5-pro"

    # Embedding Configuration
    embedding_provider: str = "google"
    embedding_model: str = "models/text-embedding-004"
    embedding_dimensions: int = 768

    # Entity Resolution
    er_vector_threshold: float = 0.70
    er_max_candidates: int = 5

    # Community Detection
    cd_max_cluster_size_factor: float = 0.5
    cd_random_seed: int = 42

    # Caching
    cache_enabled: bool = True
    cache_ttl: int = 86400  # 24 hours
    redis_host: str = "localhost"
    redis_port: int = 6379

    class Config:
        env_file = ".env"

config = KBConfig()
```

**Benefits**:
- Single source of truth
- Type-safe configuration
- Environment-specific settings
- Easy testing with mock configs

---

### 1.2 Redis Caching Layer (Effort: 3 days, Impact: High)

**Problem**: No caching for expensive operations (embeddings, LLM calls).

**Solution**: Multi-level caching with Redis.

**Implementation**:
```python
from redis import asyncio as aioredis
from functools import wraps
import json
import hashlib

class CacheManager:
    def __init__(self):
        self.redis = aioredis.from_url(
            f"redis://{config.redis_host}:{config.redis_port}"
        )

    def cache_result(self, ttl: int = 3600):
        """Decorator for caching async functions"""
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                # Generate cache key
                key = self._make_key(func.__name__, args, kwargs)

                # Try cache
                cached = await self.redis.get(key)
                if cached:
                    return json.loads(cached)

                # Compute and cache
                result = await func(*args, **kwargs)
                await self.redis.setex(key, ttl, json.dumps(result))
                return result
            return wrapper
        return decorator

    def _make_key(self, prefix: str, args, kwargs) -> str:
        """Generate cache key from function call"""
        key_str = f"{prefix}:{str(args)}:{str(kwargs)}"
        return f"kb:{hashlib.md5(key_str.encode()).hexdigest()}"

# Usage
cache = CacheManager()

class GraphIngestor:
    @cache.cache_result(ttl=86400)  # Cache for 24 hours
    async def extract(self, text: str) -> KnowledgeGraph:
        # Existing extraction logic
        pass

class EntityResolver:
    @cache.cache_result(ttl=86400)
    async def find_candidates(self, entity_name: str, embedding: List[float]):
        # Existing logic
        pass
```

**Cache Strategies**:
1. **Embedding Cache**: Cache text → embedding mappings
2. **LLM Response Cache**: Cache prompt → response pairs
3. **Similarity Cache**: Cache entity → similar entities
4. **Query Cache**: Cache query results (TTL: 1 hour)

**Expected Performance Gain**:
- Embedding operations: 80-90% cache hit rate
- Entity resolution: 60-70% cache hit rate
- Overall pipeline speedup: 3-5x for repeated entities

---

### 1.3 Hybrid Search with RRF Fusion (Effort: 3 days, Impact: High)

**Problem**: Pure vector search misses exact/keyword matches.

**Solution**: Implement Reciprocal Rank Fusion (RRF) for vector + text search.

**Implementation**:
```python
async def hybrid_entity_search(
    query_text: str,
    query_embedding: List[float],
    k: int = 10
) -> List[Dict]:
    """Hybrid search combining vector and text search with RRF"""

    async with conn.cursor() as cur:
        # Vector search results
        await cur.execute(
            """
            SELECT id, 1 - (embedding <=> %s::vector) as similarity
            FROM nodes
            ORDER BY embedding <=> %s::vector
            LIMIT 20
            """,
            (query_embedding, query_embedding)
        )
        vector_results = {str(row[0]): i+1 for i, row in enumerate(await cur.fetchall())}

        # Text search results (FTS)
        await cur.execute(
            """
            SELECT id, ts_rank_cd(to_tsvector('english', name || ' ' || COALESCE(description, '')),
                                  plainto_tsquery('english', %s)) as rank
            FROM nodes
            WHERE to_tsvector('english', name || ' ' || COALESCE(description, ''))
                  @@ plainto_tsquery('english', %s)
            ORDER BY rank DESC
            LIMIT 20
            """,
            (query_text, query_text)
        )
        text_results = {str(row[0]): i+1 for i, row in enumerate(await cur.fetchall())}

    # Reciprocal Rank Fusion
    # RRF(score) = Σ 1 / (k + rank_i)
    k_rrf = 60  # Constant (typically 50-100)
    scores = {}
    all_ids = set(vector_results.keys()) | set(text_results.keys())

    for entity_id in all_ids:
        score = 0.0
        if entity_id in vector_results:
            score += 1.0 / (k_rrf + vector_results[entity_id])
        if entity_id in text_results:
            score += 1.0 / (k_rrf + text_results[entity_id])
        scores[entity_id] = score

    # Return top k
    sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
    return sorted_results
```

**SQL Function Version**:
```sql
CREATE OR REPLACE FUNCTION hybrid_entity_search(
    query_text TEXT,
    query_vector vector(768),
    k INTEGER DEFAULT 10
) RETURNS TABLE (id UUID, score FLOAT) AS $$
    WITH vector_search AS (
        SELECT n.id,
               ROW_NUMBER() OVER (ORDER BY n.embedding <=> query_vector) as rank
        FROM nodes n
        WHERE n.embedding IS NOT NULL
        LIMIT 20
    ),
    text_search AS (
        SELECT n.id,
               ts_rank_cd(
                   to_tsvector('english', n.name || ' ' || COALESCE(n.description, '')),
                   plainto_tsquery('english', query_text)
               ) as rank
        FROM nodes n
        WHERE to_tsvector('english', n.name || ' ' || COALESCE(n.description, ''))
              @@ plainto_tsquery('english', query_text)
        LIMIT 20
    )
    SELECT COALESCE(v.id, t.id) as id,
           1.0 / (60 + COALESCE(v.rank, 1000)) +
           1.0 / (60 + COALESCE(t.rank, 1000)) as score
    FROM vector_search v
    FULL OUTER JOIN text_search t ON v.id = t.id
    ORDER BY score DESC
    LIMIT k;
$$ LANGUAGE sql STABLE;
```

**Expected Improvement**:
- Recall@10: +15-25%
- Precision@10: +10-15%
- F1 score: +12-20%

---

### 1.4 Connection Pooling (Effort: 2 days, Impact: Medium)

**Problem**: New connection created for each query (overhead ~50ms).

**Solution**: Implement async connection pool.

**Implementation**:
```python
import asyncpg
from contextlib import asynccontextmanager

class ConnectionPool:
    _instance = None
    _pool = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    async def initialize(self):
        if self._pool is None:
            self._pool = await asyncpg.create_pool(
                host=config.db_host,
                port=config.db_port,
                user=config.db_user,
                password=config.db_password,
                database=config.db_name,
                min_size=5,
                max_size=20,
                command_timeout=60
            )

    @asynccontextmanager
    async def acquire(self):
        async with self._pool.acquire() as conn:
            yield conn

    async def close(self):
        if self._pool:
            await self._pool.close()

# Usage
pool = ConnectionPool()

class EntityResolver:
    async def find_candidates(self, entity_name: str, embedding: List[float]):
        async with pool.acquire() as conn:
            async with conn.transaction():
                # Use connection
                rows = await conn.fetch(
                    "SELECT ...",
                    embedding, threshold
                )
                return rows
```

**Performance Gain**:
- Connection overhead: -95% (50ms → 2.5ms)
- Throughput: +10-20% (concurrent queries)
- Resource usage: Better under load

---

## 2. MEDIUM-TERM IMPROVEMENTS (1-3 MONTHS)

### 2.1 Incremental Updates (Effort: 2 weeks, Impact: High)

**Problem**: Full graph rebuild on each ingestion (slow for large KBs).

**Solution**: Implement incremental entity resolution and community updates.

**Strategy**:
1. **Incremental Entity Resolution**: Only resolve new entities
2. **Incremental Community Detection**: Update local clusters, global hierarchy
3. **Dirty Flag System**: Mark changed communities for re-summarization

**Implementation**:
```python
class IncrementalPipeline(KnowledgePipeline):
    async def run_incremental(self, file_path: str):
        """Process file with incremental updates"""
        graph = await self.ingestor.extract(text)

        # Only resolve new entities
        existing_names = await self._get_existing_entity_names()
        new_entities = [e for e in graph.entities if e.name not in existing_names]

        # Resolve new entities
        for entity in new_entities:
            await self.resolver.resolve_and_insert(entity)

        # Store edges and events
        await self._store_edges_and_events(graph)

        # Mark communities as dirty (don't recompute now)
        await self._mark_communities_dirty()

    async def _get_existing_entity_names(self) -> set:
        async with pool.acquire() as conn:
            rows = await conn.fetch("SELECT name FROM nodes")
            return {row['name'] for row in rows}

    async def _mark_communities_dirty(self):
        async with pool.acquire() as conn:
            await conn.execute(
                "UPDATE communities SET metadata = jsonb_set(metadata, '{dirty}', 'true')"
            )
```

**Community Update Strategy**:
```python
class IncrementalCommunityDetector:
    async def update_communities(self):
        """Update only affected communities"""
        # 1. Identify changed nodes
        dirty_nodes = await self._get_dirty_nodes()

        # 2. Identify affected communities
        affected_communities = await self._get_affected_communities(dirty_nodes)

        # 3. Rebuild only affected subgraphs
        for community_id in affected_communities:
            subgraph = await self._load_community_subgraph(community_id)
            new_memberships = self.detect_communities(subgraph)
            await self._update_community(community_id, new_memberships)

        # 4. Mark as clean
        await self._mark_clean(affected_communities)
```

**Expected Improvement**:
- Ingestion time: -70-90% for large KBs
- Resource usage: -80% (no full graph rebuild)
- Scalability: Supports 100K+ entities

---

### 2.2 Unit Testing Framework (Effort: 1 week, Impact: High)

**Problem**: Only integration test exists (slow, no component coverage).

**Solution**: Add comprehensive pytest suite with mocking.

**Implementation**:
```python
# tests/test_ingestor.py
import pytest
from unittest.mock import AsyncMock, patch
from ingestor import GraphIngestor

@pytest.fixture
def mock_llm_response():
    return KnowledgeGraph(
        entities=[
            Entity(name="Dr. Sarah Chen", type="Person", description="...")
        ],
        relationships=[
            Relationship(
                source="Dr. Sarah Chen",
                target="Project Alpha",
                type="LEADS",
                description="..."
            )
        ],
        events=[]
    )

@pytest.mark.asyncio
async def test_extract_single_entity(mock_llm_response):
    """Test entity extraction"""
    ingestor = GraphIngestor()

    with patch.object(ingestor.client.chat.completions, 'create', new=AsyncMock(return_value=mock_llm_response)):
        graph = await ingestor.extract("Dr. Sarah Chen leads Project Alpha")

    assert len(graph.entities) == 1
    assert graph.entities[0].name == "Dr. Sarah Chen"
    assert graph.entities[0].type == "Person"

@pytest.mark.asyncio
async def test_two_pass_gleaning():
    """Test two-pass extraction increases coverage"""
    # First pass response
    pass1 = KnowledgeGraph(entities=[Entity(name="Project Alpha", ...)], ...)

    # Second pass response (adds missed entity)
    pass2 = KnowledgeGraph(entities=[Entity(name="Q1 2025 Funding", ...)], ...)

    ingestor = GraphIngestor()

    with patch.object(ingestor.client.chat.completions, 'create') as mock_create:
        mock_create.side_effect = [pass1, pass2]
        graph = await ingestor.extract(text)

    assert len(graph.entities) == 2

# tests/test_resolver.py
@pytest.mark.asyncio
async def test_exact_match_returns_existing_id():
    """Test exact name match returns existing entity ID"""
    resolver = EntityResolver(db_conn_str="...")

    with patch.object(resolver, 'find_candidates') as mock_find:
        mock_find.return_value = [
            {'id': 'uuid-123', 'name': 'Dr. Sarah Chen', 'type': 'Person', ...}
        ]

        entity_id = await resolver.resolve_and_insert(
            {'name': 'Dr. Sarah Chen', 'type': 'Person', 'description': '...'},
            embedding=[0.1] * 768
        )

    assert entity_id == 'uuid-123'

@pytest.mark.asyncio
async def test_normalized_match_handles_titles():
    """Test name normalization removes titles"""
    resolver = EntityResolver(db_conn_str="...")
    assert resolver._normalize_name("Dr. Sarah Chen") == "sarah chen"
    assert resolver._normalize_name("Director Samuel Oakley") == "samuel oakley"

# tests/test_community.py
@pytest.mark.asyncio
async def test_leiden_creates_connected_communities():
    """Test Leiden algorithm produces connected communities"""
    detector = CommunityDetector(db_conn_str="...")
    G = nx.Graph()
    G.add_edges_from([(1,2), (2,3), (3,4), (5,6)])

    memberships = detector.detect_communities(G)

    # Check no disconnected nodes in same community
    for m in memberships:
        cluster_nodes = [m2['node_id'] for m2 in memberships
                       if m2['cluster_id'] == m['cluster_id']]
        subgraph = G.subgraph(cluster_nodes)
        assert nx.is_connected(subgraph)
```

**Coverage Goals**:
- Ingestor: 90%+ coverage
- Resolver: 85%+ coverage
- Community: 80%+ coverage
- Summarizer: 75%+ coverage
- Pipeline: 70%+ coverage

---

### 2.3 Performance Monitoring (Effort: 3 days, Impact: Medium)

**Problem**: No visibility into system performance and bottlenecks.

**Solution**: Add Prometheus metrics and Grafana dashboards.

**Implementation**:
```python
from prometheus_client import Counter, Histogram, Gauge
import time

# Metrics
entity_resolution_time = Histogram(
    'kb_entity_resolution_seconds',
    'Time to resolve entity',
    ['decision']  # MERGE, INSERT, LINK
)

embedding_generation_time = Histogram(
    'kb_embedding_generation_seconds',
    'Time to generate embedding'
)

community_detection_time = Histogram(
    'kb_community_detection_seconds',
    'Time for community detection',
    ['graph_size']  # small, medium, large
)

cache_hits = Counter(
    'kb_cache_hits_total',
    'Cache hits',
    ['cache_type']  # embedding, llm, similarity
)

cache_misses = Counter(
    'kb_cache_misses_total',
    'Cache misses',
    ['cache_type']
)

total_entities = Gauge('kb_total_entities', 'Total entities in database')
total_communities = Gauge('kb_total_communities', 'Total communities')
hierarchy_depth = Gauge('kb_hierarchy_depth', 'Depth of community hierarchy')

class MonitoredEntityResolver(EntityResolver):
    async def resolve_and_insert(self, entity, embedding):
        start = time.time()

        result = await super().resolve_and_insert(entity, embedding)

        duration = time.time() - start

        # Determine decision type
        if result == 'insert':
            decision = 'INSERT'
        else:
            decision = 'MERGE'  # Simplified

        entity_resolution_time.labels(decision=decision).observe(duration)

        return result

class MonitoredSummarizer(CommunitySummarizer):
    async def summarize_all(self):
        await super().summarize_all()

        # Update gauges
        async with pool.acquire() as conn:
            total_entities.set(await conn.fetchval("SELECT COUNT(*) FROM nodes"))
            total_communities.set(await conn.fetchval("SELECT COUNT(*) FROM communities"))
            hierarchy_depth.set(await conn.fetchval("SELECT MAX(level) FROM communities"))
```

**Prometheus Configuration**:
```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'knowledge_base'
    static_configs:
      - targets: ['localhost:8000']
    scrape_interval: 15s
```

**Grafana Dashboards**:
- Pipeline performance (latency, throughput)
- Cache effectiveness (hit/miss ratios)
- Entity resolution metrics (merge/insert rates)
- Community metrics (depth, size distribution)
- Database performance (query latency, connection pool)

---

### 2.4 Intelligent Context Chunking (Effort: 1 week, Impact: Medium)

**Problem**: Simple truncation loses information for large communities.

**Solution**: Semantic chunking with importance ranking.

**Implementation**:
```python
from sentence_transformers import SentenceTransformer
import numpy as np

class SemanticChunker:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def chunk_text(self, text: str, max_tokens: int = 4000) -> list[str]:
        """Split text into semantic chunks"""

        # Split into sentences
        sentences = text.split('. ')

        # Get embeddings
        embeddings = self.model.encode(sentences)

        # Create chunks by combining adjacent sentences
        chunks = []
        current_chunk = []
        current_tokens = 0

        for sentence, embedding in zip(sentences, embeddings):
            sentence_tokens = len(sentence.split())

            if current_tokens + sentence_tokens > max_tokens and current_chunk:
                chunks.append('. '.join(current_chunk))
                current_chunk = [sentence]
                current_tokens = sentence_tokens
            else:
                current_chunk.append(sentence)
                current_tokens += sentence_tokens

        if current_chunk:
            chunks.append('. '.join(current_chunk))

        return chunks

    def rank_chunks_by_query(self, chunks: list[str], query: str, top_k: int = 3) -> list[str]:
        """Rank chunks by relevance to query"""

        query_embedding = self.model.encode([query])
        chunk_embeddings = self.model.encode(chunks)

        # Compute cosine similarity
        similarities = np.dot(chunk_embeddings, query_embedding.T).flatten()

        # Get top-k
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        return [chunks[i] for i in top_indices]

# Usage in summarizer
class SmartSummarizer(CommunitySummarizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.chunker = SemanticChunker()

    async def _gather_context(self, community_id: str, level: int) -> str:
        # Gather full context (no LIMIT)
        context = await self._get_full_context(community_id, level)

        # Chunk semantically
        chunks = self.chunker.chunk_text(context, max_tokens=3000)

        # Rank by query (if applicable)
        if self.query:
            chunks = self.chunker.rank_chunks_by_query(chunks, self.query, top_k=3)

        return ' '.join(chunks)
```

**Expected Improvement**:
- Information retention: +20-30%
- Coherence: +15-20%
- Summarization quality: +10-15%

---

## 3. LONG-TERM STRATEGIC INITIATIVES (3-12 MONTHS)

### 3.1 Graph Neural Networks for Entity Embeddings
**Effort**: 6-8 weeks | **Impact**: Very High | **Priority**: HIGH

**Problem**: Pure text embeddings miss structural/relational information.

**Solution**: Use GNNs to learn embeddings from graph topology.

**See Section 4.1 for detailed plan**

---

### 3.2 Automated Knowledge Graph Completion
**Effort**: 6-10 weeks | **Impact**: Very High | **Priority**: HIGH

**Problem**: Manual extraction misses implicit relationships and entities.

**Solution**: Use ML models to predict missing edges and nodes.

**See Section 4.2 for detailed plan**

---

### 3.3 Multi-Modal Support (Effort: 8-12 weeks, Impact: High)

**Problem**: Current system only processes text.

**Solution**: Support images, PDFs, audio, video.

**Implementation**:
```python
class MultiModalIngestor:
    def __init__(self):
        self.text_ingestor = GraphIngestor()
        self.image_processor = CLIPProcessor()  # For image embeddings
        self.audio_processor = WhisperProcessor()  # For audio transcription

    async def extract_multimodal(self, input_data: Union[str, bytes, dict]):
        """Extract knowledge from multiple modalities"""

        if isinstance(input_data, str) and input_data.endswith('.txt'):
            # Text
            return await self.text_ingestor.extract(input_data)

        elif isinstance(input_data, bytes) and self._is_image(input_data):
            # Image
            image_embedding = self.image_processor.embed(input_data)
            image_text = self.image_processor.describe(input_data)

            # Extract entities from image description
            return await self.text_ingestor.extract(image_text)

        elif isinstance(input_data, str) and input_data.endswith('.pdf'):
            # PDF
            text = self._extract_pdf_text(input_data)
            return await self.text_ingestor.extract(text)

        elif isinstance(input_data, dict) and 'audio' in input_data:
            # Audio
            transcript = self.audio_processor.transcribe(input_data['audio'])
            return await self.text_ingestor.extract(transcript)

    async def _extract_pdf_text(self, pdf_path: str) -> str:
        """Extract text from PDF"""
        import PyPDF2
        with open(pdf_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            text = ''
            for page in reader.pages:
                text += page.extract_text()
        return text
```

**Multi-Modal Embeddings**:
```python
# Entity with multi-modal embeddings
class MultiModalEntity(BaseModel):
    id: str
    name: str
    type: str
    description: str
    text_embedding: List[float]  # 768D
    image_embedding: Optional[List[float]]  # 512D
    audio_embedding: Optional[List[float]]  # 512D
    metadata: dict
```

---

### 3.4 Federated Learning for Privacy (Effort: 10-16 weeks, Impact: High)

**Problem**: Cannot share data across organizations for privacy reasons.

**Solution**: Federated learning to train models without data sharing.

**Architecture**:
```
Organization A              Organization B              Organization C
     │                            │                           │
     ├─ Local Training           ├─ Local Training          ├─ Local Training
     │                            │                           │
     └─────────┬──────────────────┴───────────┬───────────────┘
               │                              │
               ▼                              ▼
         Federated Server (Aggregates Updates)
               │
               ▼
         Global Model (Distributed to all)
```

---

## 4. PRIORITY MATRIX

| Initiative | Impact | Effort | Priority | Timeline |
|------------|--------|--------|----------|----------|
| Configuration Management | High | Low | **P0** | Week 1 |
| Redis Caching | High | Low | **P0** | Week 2 |
| Hybrid Search (RRF) | High | Low | **P0** | Week 2 |
| Connection Pooling | Medium | Low | **P1** | Week 3 |
| Unit Testing | High | Medium | **P0** | Month 1 |
| Performance Monitoring | Medium | Low | **P1** | Month 1 |
| Incremental Updates | High | Medium | **P0** | Month 2 |
| Semantic Chunking | Medium | Medium | **P1** | Month 2 |
| **GNN Entity Embeddings** | **Very High** | **High** | **P0** | **Month 3-4** |
| **KG Completion** | **Very High** | **High** | **P0** | **Month 4-5** |
| Multi-Modal Support | High | High | P1 | Month 6-8 |
| Federated Learning | High | Very High | P2 | Month 9-12 |

---

## 5. IMPLEMENTATION DEPENDENCIES

```
Configuration Management
    ├─ Required for: ALL other improvements
    └─ Enables: Testing, monitoring, tuning

Redis Caching
    ├─ Improves: Embedding generation, entity resolution
    └─ Prerequisite for: High-scale deployment

Unit Testing
    ├─ Required for: All future features
    └─ Enables: Refactoring, confidence in changes

Incremental Updates
    ├─ Requires: Unit tests (for stability)
    ├─ Enables: Large-scale KB (100K+ entities)
    └─ Prerequisite for: Real-time updates

GNN Entity Embeddings
    ├─ Requires: Graph export, training pipeline
    ├─ Improves: Entity resolution, community detection
    └─ Prerequisite for: KG completion

KG Completion
    ├─ Requires: GNN embeddings (for link prediction)
    ├─ Improves: Knowledge coverage
    └─ Synergy with: Entity resolution, community detection
```

---

## 6. ROLLBACK STRATEGIES

For each major feature, implement feature flags for safe rollback:

```python
# Feature flags
ENABLE_GNN_EMBEDDINGS = os.getenv("ENABLE_GNN", "false").lower() == "true"
ENABLE_KG_COMPLETION = os.getenv("ENABLE_KG_COMPLETION", "false").lower() == "true"
ENABLE_INCREMENTAL = os.getenv("ENABLE_INCREMENTAL", "true").lower() == "true"

class EntityResolver:
    async def resolve_and_insert(self, entity, embedding):
        if ENABLE_GNN_EMBEDDINGS:
            # Use GNN embeddings
            gnn_embedding = await self._get_gnn_embedding(entity['name'])
            return await self._resolve_with_gnn(entity, gnn_embedding)
        else:
            # Fall back to text embeddings
            return await super().resolve_and_insert(entity, embedding)
```

---

## 7. SUCCESS METRICS

**Performance Targets** (after Quick Wins):
- Pipeline latency: <30 seconds for 10-page document
- Cache hit rate: >80%
- Query latency: <100ms (hybrid search)
- Throughput: 100 documents/hour

**Quality Targets** (after GNN + KG Completion):
- Entity resolution F1: >0.95
- Link prediction accuracy: >0.85
- Community modularity: >0.6
- Summarization coherence (human rating): >4/5

---

## 8. NEXT STEPS

**Week 1**:
- Implement configuration management
- Add connection pooling
- Set up pytest framework

**Week 2-3**:
- Implement Redis caching
- Add hybrid search (RRF)
- Write unit tests for core components

**Month 2**:
- Implement incremental updates
- Add performance monitoring
- Deploy to staging environment

**Month 3-4**:
- Design and implement GNN entity embeddings
- Benchmark against baseline

**Month 4-5**:
- Implement KG completion
- Evaluate and iterate

---

**Document Status**: ✅ COMPLETE
**Next Document**: 02_RESEARCH_IDEAS.md
