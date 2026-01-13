# AUTONOMOUS AGENT - TASK BREAKDOWN

**Version**: 1.0
**Date**: January 13, 2026
**Status**: Ready for Implementation

---

## PHASE 1: FOUNDATION (WEEK 1) - 7 TASKS

### Task 1.1: Project Structure Setup
**Priority**: P0 | **Effort**: 2 hours | **Dependencies**: None

**Description**:
Create the research_agent directory structure with all necessary subdirectories.

**Actions**:
```bash
cd unbiased-observer
mkdir -p research_agent/{orchestrator,research,ingestion,processing,monitoring}
touch research_agent/{__init__.py,main.py}
touch research_agent/orchestrator/{__init__.py,scheduler.py,task_queue.py,error_recovery.py}
touch research_agent/research/{__init__.py,source_discovery.py,content_fetcher.py,content_extractor.py}
touch research_agent/ingestion/{__init__.py,pipeline.py,async_ingestor.py,postgres_storage.py}
touch research_agent/processing/{__init__.py,coordinator.py,trigger.py}
touch research_agent/monitoring/{__init__.py,metrics.py,health_checker.py}
```

**Success Criteria**:
- [ ] All directories created
- [ ] All __init__.py files present
- [ ] Project structure matches plan

**Verification**: `tree research_agent/` shows correct structure

---

### Task 1.2: Configuration Management
**Priority**: P0 | **Effort**: 3 hours | **Dependencies**: 1.1

**Description**:
Implement configuration loading from YAML files with environment variable override support.

**File**: `research_agent/config.py`

**Implementation**:
```python
import os
import yaml
from pathlib import Path
from typing import Dict, Any
from dataclasses import dataclass

@dataclass
class DatabaseConfig:
    connection_string: str
    pool_min_size: int = 5
    pool_max_size: int = 20

@dataclass
class LLMConfig:
    base_url: str
    api_key: str
    model_default: str
    model_pro: str
    max_retries: int = 3
    timeout: int = 120

@dataclass
class EmbeddingConfig:
    provider: str
    model: str
    api_key_env: str
    dimensions: int = 768

@dataclass
class ResearchConfig:
    sources_config: str
    max_concurrent_fetches: int = 10
    rate_limit: float = 2.0
    max_content_length: int = 1000000

@dataclass
class IngestionConfig:
    max_concurrent_llm_calls: int = 3
    batch_size: int = 10
    retry_backoff_factor: float = 2.0
    max_retries: int = 3

@dataclass
class ProcessingConfig:
    min_entities_to_process: int = 100
    min_time_between_processing_hours: int = 1
    processing_interval_seconds: int = 60

@dataclass
class MonitoringConfig:
    metrics_port: int = 8000
    health_check_interval_seconds: int = 300
    log_retention_days: int = 30

@dataclass
class PathsConfig:
    knowledge_base: str
    cache_dir: str = "./cache"
    logs_dir: str = "./logs"
    state_dir: str = "./state"

class Config:
    """Main configuration class"""

    def __init__(self, config_path: str = "research_agent_config.yaml"):
        self.config_path = config_path
        self._load_config()

    def _load_config(self):
        """Load configuration from YAML file"""

        with open(self.config_path, 'r') as f:
            config_data = yaml.safe_load(f)

        # Parse sections
        self.database = DatabaseConfig(**config_data.get('database', {}))
        self.llm = LLMConfig(**config_data.get('llm', {}))
        self.embedding = EmbeddingConfig(**config_data.get('embedding', {}))
        self.research = ResearchConfig(**config_data.get('research', {}))
        self.ingestion = IngestionConfig(**config_data.get('ingestion', {}))
        self.processing = ProcessingConfig(**config_data.get('processing', {}))
        self.monitoring = MonitoringConfig(**config_data.get('monitoring', {}))
        self.paths = PathsConfig(**config_data.get('paths', {}))

        # Override with environment variables
        self._apply_env_overrides()

    def _apply_env_overrides(self):
        """Override config with environment variables"""

        if 'DB_CONNECTION_STRING' in os.environ:
            self.database.connection_string = os.environ['DB_CONNECTION_STRING']

        if 'LLM_BASE_URL' in os.environ:
            self.llm.base_url = os.environ['LLM_BASE_URL']

        if 'LLM_API_KEY' in os.environ:
            self.llm.api_key = os.environ['LLM_API_KEY']

        if 'GOOGLE_API_KEY' in os.environ:
            os.embedding.api_key_env = 'GOOGLE_API_KEY'

# Global config instance
config = None

def load_config(config_path: str = None) -> Config:
    """Load and return configuration"""

    global config
    if config is None:
        config = Config(config_path or "configs/research_agent_config.yaml")
    return config
```

**Success Criteria**:
- [ ] Config class loads YAML correctly
- [ ] Environment variable overrides work
- [ ] All config sections accessible
- [ ] Default values applied

**Verification**: Write test to verify config loading

---

### Task 1.3: Database Schema Extension
**Priority**: P0 | **Effort**: 2 hours | **Dependencies**: None

**Description**:
Create additional tables in knowledge_graph database for task queue and agent state.

**File**: `research_agent/db_schema.sql`

**Implementation**:
```sql
-- ================================================================
-- AUTONOMOUS AGENT DATABASE SCHEMA
-- ================================================================

-- 1. Task Queue
CREATE TABLE IF NOT EXISTS research_tasks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    type TEXT NOT NULL,  -- 'FETCH', 'INGEST', 'PROCESS'
    source TEXT,  -- URL, file path, etc.
    metadata JSONB DEFAULT '{}',
    status TEXT DEFAULT 'PENDING',  -- 'PENDING', 'IN_PROGRESS', 'COMPLETED', 'FAILED'
    worker_id TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    retry_count INTEGER DEFAULT 0,
    max_retries INTEGER DEFAULT 3,
    error_message TEXT,
    error_details JSONB DEFAULT '{}',
    CONSTRAINT chk_status CHECK (status IN ('PENDING', 'IN_PROGRESS', 'COMPLETED', 'FAILED'))
);

CREATE INDEX IF NOT EXISTS idx_tasks_status ON research_tasks(status);
CREATE INDEX IF NOT EXISTS idx_tasks_worker ON research_tasks(worker_id);
CREATE INDEX IF NOT EXISTS idx_tasks_created ON research_tasks(created_at);

-- 2. Agent State
CREATE TABLE IF NOT EXISTS agent_state (
    key TEXT PRIMARY KEY,
    value JSONB NOT NULL,
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_state_updated ON agent_state(updated_at);

-- 3. Research Sources
CREATE TABLE IF NOT EXISTS research_sources (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    type TEXT NOT NULL,  -- 'manual', 'rss', 'api'
    name TEXT NOT NULL,
    url TEXT,
    description TEXT,
    config JSONB DEFAULT '{}',
    is_active BOOLEAN DEFAULT TRUE,
    last_fetched_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_sources_active ON research_sources(is_active);
CREATE INDEX IF NOT EXISTS idx_sources_type ON research_sources(type);

-- 4. Ingestion Logs
CREATE TABLE IF NOT EXISTS ingestion_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    source TEXT NOT NULL,
    task_id UUID REFERENCES research_tasks(id),
    status TEXT NOT NULL,
    entities_stored INTEGER DEFAULT 0,
    edges_stored INTEGER DEFAULT 0,
    events_stored INTEGER DEFAULT 0,
    duration_seconds FLOAT,
    error_message TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_logs_task ON ingestion_logs(task_id);
CREATE INDEX IF NOT EXISTS idx_logs_status ON ingestion_logs(status);
CREATE INDEX IF NOT EXISTS idx_logs_created ON ingestion_logs(created_at);

-- 5. Processing Logs
CREATE TABLE IF NOT EXISTS processing_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    task_id UUID REFERENCES research_tasks(id),
    status TEXT NOT NULL,
    nodes_processed INTEGER DEFAULT 0,
    communities_created INTEGER DEFAULT 0,
    duration_seconds FLOAT,
    error_message TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_processing_task ON processing_logs(task_id);
CREATE INDEX IF NOT EXISTS idx_processing_created ON processing_logs(created_at);
```

**Success Criteria**:
- [ ] All tables created successfully
- [ ] Indexes created
- [ ] Constraints applied
- [ ] No errors in schema

**Verification**: Run schema against database, verify tables exist

---

### Task 1.4: Logging Setup
**Priority**: P0 | **Effort**: 2 hours | **Dependencies**: 1.2

**Description**:
Set up structured logging with file rotation and multiple handlers.

**File**: `research_agent/__init__.py` (logging setup)

**Implementation**:
```python
import logging
import logging.handlers
from pathlib import Path
from datetime import datetime

def setup_logging(config: 'Config'):
    """Setup structured logging for agent"""

    # Create logs directory
    logs_dir = Path(config.paths.logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(config.log_level or 'INFO')

    # Formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s'
    )
    json_formatter = logging.Formatter(
        '{"timestamp": "%(asctime)s", "level": "%(levelname)s", '
        '"logger": "%(name)s", "line": %(lineno)d, "message": "%(message)s"}'
    )

    # File handlers with rotation
    # Main agent log
    agent_handler = logging.handlers.RotatingFileHandler(
        logs_dir / 'agent.log',
        maxBytes=10*1024*1024,  # 10MB
        backupCount=10
    )
    agent_handler.setFormatter(detailed_formatter)
    agent_handler.setLevel('INFO')

    # Error log
    error_handler = logging.handlers.RotatingFileHandler(
        logs_dir / 'agent_error.log',
        maxBytes=10*1024*1024,
        backupCount=10
    )
    error_handler.setFormatter(detailed_formatter)
    error_handler.setLevel('ERROR')

    # Ingestion log
    ingestion_handler = logging.handlers.RotatingFileHandler(
        logs_dir / 'ingestion.log',
        maxBytes=10*1024*1024,
        backupCount=10
    )
    ingestion_handler.setFormatter(detailed_formatter)
    ingestion_handler.setLevel('INFO')

    # Processing log
    processing_handler = logging.handlers.RotatingFileHandler(
        logs_dir / 'processing.log',
        maxBytes=10*1024*1024,
        backupCount=10
    )
    processing_handler.setFormatter(detailed_formatter)
    processing_handler.setLevel('INFO')

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(detailed_formatter)
    console_handler.setLevel('INFO')

    # Add handlers
    root_logger.addHandler(agent_handler)
    root_logger.addHandler(error_handler)
    root_logger.addHandler(ingestion_handler)
    root_logger.addHandler(processing_handler)
    root_logger.addHandler(console_handler)

    # Module loggers
    agent_logger = logging.getLogger('research_agent')
    ingestion_logger = logging.getLogger('research_agent.ingestion')
    processing_logger = logging.getLogger('research_agent.processing')

    return agent_logger, ingestion_logger, processing_logger

# Get logger instance
def get_logger(name: str):
    """Get logger instance"""
    return logging.getLogger(name)
```

**Success Criteria**:
- [ ] All log files created
- [ ] Rotation configured
- [ ] Console output works
- [ ] Different log levels for different modules

**Verification**: Run agent, check logs directory

---

### Task 1.5: Task Queue Implementation
**Priority**: P0 | **Effort**: 4 hours | **Dependencies**: 1.2, 1.3

**Description**:
Implement persistent task queue with PostgreSQL backend.

**File**: `research_agent/orchestrator/task_queue.py`

**Implementation**:
```python
import psycopg
from typing import List, Optional
from dataclasses import dataclass
from datetime import datetime
import logging

logger = logging.getLogger('research_agent.orchestrator')

@dataclass
class ResearchTask:
    id: str
    type: str
    source: Optional[str]
    metadata: dict
    status: str
    worker_id: Optional[str]
    created_at: datetime
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    retry_count: int
    max_retries: int
    error_message: Optional[str]

class TaskQueue:
    """Persistent task queue with PostgreSQL backend"""

    def __init__(self, db_conn_str: str):
        self.db_conn_str = db_conn_str

    async def initialize(self):
        """Initialize connection pool"""
        self.pool = await psycopg.AsyncConnectionPool(
            self.db_conn_str,
            min_size=5,
            max_size=20
        )
        logger.info("Task queue initialized")

    async def add_task(
        self,
        task_type: str,
        source: str = None,
        metadata: dict = None
    ) -> str:
        """Add new task to queue"""

        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    INSERT INTO research_tasks (type, source, metadata)
                    VALUES (%s, %s, %s)
                    RETURNING id
                    """,
                    (task_type, source, psycopg.types.Json(metadata or {}))
                )
                row = await cur.fetchone()
                task_id = str(row[0])

        logger.info(f"Task added: {task_id} ({task_type})")
        return task_id

    async def get_next_task(
        self,
        worker_id: str,
        task_types: List[str] = None
    ) -> Optional[ResearchTask]:
        """Get next pending task"""

        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                # Build type filter
                type_filter = ""
                params = [worker_id]
                if task_types:
                    placeholders = ','.join(['%s'] * len(task_types))
                    type_filter = f"AND type IN ({placeholders})"
                    params.extend(task_types)

                # Get and lock task
                await cur.execute(
                    f"""
                    UPDATE research_tasks
                    SET status = 'IN_PROGRESS',
                        worker_id = %s,
                        started_at = NOW()
                    WHERE id = (
                        SELECT id FROM research_tasks
                        WHERE status = 'PENDING'
                        {type_filter}
                        ORDER BY created_at ASC
                        LIMIT 1
                        FOR UPDATE SKIP LOCKED
                    )
                    RETURNING id, type, source, metadata, status,
                              worker_id, created_at, started_at,
                              completed_at, retry_count, max_retries
                    """,
                    params
                )

                row = await cur.fetchone()

                if not row:
                    return None

                task = ResearchTask(
                    id=str(row[0]),
                    type=row[1],
                    source=row[2],
                    metadata=row[3],
                    status=row[4],
                    worker_id=row[5],
                    created_at=row[6],
                    started_at=row[7],
                    completed_at=row[8],
                    retry_count=row[9],
                    max_retries=row[10]
                )

        logger.info(f"Task claimed: {task.id} by {worker_id}")
        return task

    async def update_task_status(
        self,
        task_id: str,
        status: str,
        error_message: str = None,
        error_details: dict = None
    ):
        """Update task status"""

        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    UPDATE research_tasks
                    SET status = %s,
                        completed_at = CASE WHEN %s = 'COMPLETED' OR %s = 'FAILED' THEN NOW() ELSE NULL END,
                        error_message = %s,
                        error_details = %s
                    WHERE id = %s
                    """,
                    (status, status, status, error_message, psycopg.types.Json(error_details or {}), task_id)
                )

        logger.info(f"Task updated: {task_id} -> {status}")

    async def increment_retry(self, task_id: str):
        """Increment retry count for failed task"""

        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    UPDATE research_tasks
                    SET retry_count = retry_count + 1,
                        status = CASE WHEN retry_count + 1 >= max_retries THEN 'FAILED' ELSE 'PENDING' END,
                        error_message = NULL,
                        error_details = '{}'
                    WHERE id = %s
                    RETURNING status, retry_count, max_retries
                    """,
                    (task_id,)
                )

                row = await cur.fetchone()
                new_status = row[0]

        logger.info(f"Task retry: {task_id} -> {new_status}")

    async def get_failed_tasks(
        self,
        retry_limit: int = 3
    ) -> List[ResearchTask]:
        """Get tasks that failed but are retryable"""

        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    SELECT id, type, source, metadata, status,
                           worker_id, created_at, started_at,
                           completed_at, retry_count, max_retries,
                           error_message
                    FROM research_tasks
                    WHERE status = 'FAILED'
                    AND retry_count < %s
                    ORDER BY created_at ASC
                    LIMIT 100
                    """,
                    (retry_limit,)
                )

                rows = await cur.fetchall()
                tasks = [
                    ResearchTask(
                        id=str(row[0]), type=row[1], source=row[2],
                        metadata=row[3], status=row[4], worker_id=row[5],
                        created_at=row[6], started_at=row[7], completed_at=row[8],
                        retry_count=row[9], max_retries=row[10], error_message=row[11]
                    )
                    for row in rows
                ]

        logger.info(f"Found {len(tasks)} retryable tasks")
        return tasks

    async def get_pending_count(self, task_type: str = None) -> int:
        """Get count of pending tasks"""

        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                if task_type:
                    await cur.execute(
                        "SELECT COUNT(*) FROM research_tasks WHERE status = 'PENDING' AND type = %s",
                        (task_type,)
                    )
                else:
                    await cur.execute(
                        "SELECT COUNT(*) FROM research_tasks WHERE status = 'PENDING'"
                    )

                row = await cur.fetchone()
                return row[0]

    async def close(self):
        """Close connection pool"""
        await self.pool.close()
        logger.info("Task queue closed")
```

**Success Criteria**:
- [ ] Tasks can be added to queue
- [ ] Workers can claim tasks (FOR UPDATE)
- [ ] Status updates work correctly
- [ ] Retry mechanism works

**Verification**: Write integration test for task queue

---

### Task 1.6: Error Recovery Implementation
**Priority**: P1 | **Effort**: 3 hours | **Dependencies**: 1.5

**Description**:
Implement exponential backoff retry mechanism for failed operations.

**File**: `research_agent/orchestrator/error_recovery.py`

**Implementation**:
```python
import asyncio
import logging
from typing import Callable, Optional
from functools import wraps

logger = logging.getLogger('research_agent.orchestrator')

class MaxRetriesExceeded(Exception):
    """Maximum retries exceeded"""
    pass

class ErrorRecovery:
    """Error recovery with exponential backoff"""

    def __init__(
        self,
        max_retries: int = 3,
        backoff_factor: float = 2.0,
        base_delay: float = 1.0
    ):
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.base_delay = base_delay

    async def execute_with_retry(
        self,
        func: Callable,
        *args,
        error_types: tuple = (Exception,),
        **kwargs
    ):
        """
        Execute function with exponential backoff retry

        Args:
            func: Async function to execute
            error_types: Tuple of exception types to catch
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Result of function

        Raises:
            MaxRetriesExceeded: If all retries fail
        """

        last_exception = None

        for attempt in range(self.max_retries):
            try:
                return await func(*args, **kwargs)

            except error_types as e:
                last_exception = e

                if attempt == self.max_retries - 1:
                    # Last attempt failed
                    logger.error(f"All {self.max_retries} attempts failed: {e}")
                    raise MaxRetriesExceeded(f"Function failed after {self.max_retries} retries") from e

                # Calculate backoff delay
                delay = self.base_delay * (self.backoff_factor ** attempt)

                logger.warning(
                    f"Attempt {attempt+1}/{self.max_retries} failed: {e}. "
                    f"Retrying in {delay:.2f}s"
                )

                await asyncio.sleep(delay)

        # Should not reach here
        raise MaxRetriesExceeded("Unexpected error recovery state")

def with_retry(
    max_retries: int = 3,
    backoff_factor: float = 2.0,
    base_delay: float = 1.0,
    error_types: tuple = (Exception,)
):
    """
    Decorator for automatic retry with exponential backoff

    Usage:
        @with_retry(max_retries=3)
        async def my_function():
            # Your code here
    """

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            recovery = ErrorRecovery(max_retries, backoff_factor, base_delay)
            return await recovery.execute_with_retry(
                func,
                *args,
                error_types=error_types,
                **kwargs
            )
        return wrapper
    return decorator
```

**Success Criteria**:
- [ ] Exponential backoff works
- [ ] Max retries enforced
- [ ] Error logged correctly
- [ ] Works with any async function

**Verification**: Write test with failing function, verify retry behavior

---

### Task 1.7: Scheduler Implementation
**Priority**: P0 | **Effort**: 4 hours | **Dependencies**: 1.5, 1.6

**Description**:
Implement APScheduler for 24/7 task processing and scheduling.

**File**: `research_agent/orchestrator/scheduler.py`

**Implementation**:
```python
import asyncio
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.triggers.cron import CronTrigger
import logging
from typing import Dict, Any

logger = logging.getLogger('research_agent.orchestrator')

class AgentScheduler:
    """24/7 scheduler for autonomous research agent"""

    def __init__(self, task_queue: 'TaskQueue', config: 'Config'):
        self.scheduler = AsyncIOScheduler()
        self.task_queue = task_queue
        self.config = config
        self.worker_id = f"worker-{asyncio.get_event_loop().time()}"

        # Job references
        self.jobs: Dict[str, Any] = {}

    async def start(self):
        """Start scheduler with all jobs"""

        # Job 1: Process task queue (every 10 seconds)
        self.jobs['process_queue'] = self.scheduler.add_job(
            self._process_queue,
            trigger=IntervalTrigger(seconds=10),
            id='process_queue',
            name='Process Task Queue',
            max_instances=1
        )

        # Job 2: Processing pipeline (every 60 seconds)
        self.jobs['processing_pipeline'] = self.scheduler.add_job(
            self._process_ingestion_queue,
            trigger=IntervalTrigger(seconds=60),
            id='processing_pipeline',
            name='Processing Pipeline',
            max_instances=1
        )

        # Job 3: Monitoring (every 5 minutes)
        self.jobs['monitoring'] = self.scheduler.add_job(
            self._monitoring,
            trigger=IntervalTrigger(seconds=300),
            id='monitoring',
            name='Health Monitoring',
            max_instances=1
        )

        # Job 4: Retry failed tasks (every 5 minutes)
        self.jobs['retry_failed'] = self.scheduler.add_job(
            self._retry_failed_tasks,
            trigger=IntervalTrigger(seconds=300),
            id='retry_failed',
            name='Retry Failed Tasks',
            max_instances=1
        )

        # Start scheduler
        self.scheduler.start()

        logger.info("Scheduler started with 4 jobs:")
        for job_id, job in self.jobs.items():
            logger.info(f"  - {job_id}: {job.name}")

    async def stop(self):
        """Stop scheduler gracefully"""

        logger.info("Stopping scheduler...")

        # Remove all jobs
        for job_id in list(self.jobs.keys()):
            self.scheduler.remove_job(job_id)

        # Shutdown scheduler
        self.scheduler.shutdown(wait=True)

        logger.info("Scheduler stopped")

    async def _process_queue(self):
        """Process research tasks from queue"""

        try:
            # Get next task
            task = await self.task_queue.get_next_task(
                self.worker_id,
                task_types=['FETCH', 'INGEST']
            )

            if not task:
                logger.debug("No pending tasks in queue")
                return

            logger.info(f"Processing task: {task.type} - {task.source}")

            # Route to appropriate handler
            if task.type == 'FETCH':
                await self._handle_fetch_task(task)
            elif task.type == 'INGEST':
                await self._handle_ingest_task(task)

            # Mark task as completed
            await self.task_queue.update_task_status(task.id, 'COMPLETED')

        except Exception as e:
            logger.error(f"Error processing task: {e}", exc_info=True)
            # Mark as failed (will be retried)
            await self.task_queue.increment_retry(task.id)

    async def _process_ingestion_queue(self):
        """Process ingested content through pipeline"""

        try:
            # Get next task
            task = await self.task_queue.get_next_task(
                self.worker_id,
                task_types=['PROCESS']
            )

            if not task:
                return

            # Processing pipeline logic (to be implemented in Phase 4)
            await self._handle_process_task(task)

            await self.task_queue.update_task_status(task.id, 'COMPLETED')

        except Exception as e:
            logger.error(f"Error in processing: {e}", exc_info=True)
            await self.task_queue.increment_retry(task.id)

    async def _retry_failed_tasks(self):
        """Retry failed tasks that haven't exceeded retries"""

        try:
            failed_tasks = await self.task_queue.get_failed_tasks()

            for task in failed_tasks:
                logger.info(f"Retrying task: {task.id}")

                # Reset to PENDING
                await self.task_queue.update_task_status(
                    task.id,
                    'PENDING',
                    error_message=None
                )

        except Exception as e:
            logger.error(f"Error retrying failed tasks: {e}", exc_info=True)

    async def _monitoring(self):
        """Health checks and metrics collection"""

        try:
            logger.debug("Running health checks...")

            # Check task queue
            pending_count = await self.task_queue.get_pending_count()
            logger.info(f"Pending tasks: {pending_count}")

            # Check database connectivity
            # (Implement in monitoring phase)

            # Check LLM API
            # (Implement in monitoring phase)

        except Exception as e:
            logger.error(f"Error in monitoring: {e}", exc_info=True)

    async def _handle_fetch_task(self, task: 'ResearchTask'):
        """Handle content fetch task"""
        # Placeholder - will be implemented in Phase 2
        logger.info(f"Fetching content from: {task.source}")
        pass

    async def _handle_ingest_task(self, task: 'ResearchTask'):
        """Handle ingestion task"""
        # Placeholder - will be implemented in Phase 3
        logger.info(f"Ingesting content: {task.id}")
        pass

    async def _handle_process_task(self, task: 'ResearchTask'):
        """Handle processing task"""
        # Placeholder - will be implemented in Phase 4
        logger.info(f"Processing: {task.id}")
        pass
```

**Success Criteria**:
- [ ] Scheduler starts without errors
- [ ] All 4 jobs scheduled correctly
- [ ] Task queue processing works
- [ ] Graceful shutdown works

**Verification**: Run scheduler for 1 minute, verify all jobs executed

---

## PHASE 2: RESEARCH AGENT (WEEK 2) - 5 TASKS

### Task 2.1: Content Fetcher
**Priority**: P0 | **Effort**: 4 hours | **Dependencies**: 1.6

**File**: `research_agent/research/content_fetcher.py`

**Implementation**:
```python
import aiohttp
import asyncio
import logging
from typing import Dict, List
from research_agent.orchestrator.error_recovery import with_retry, MaxRetriesExceeded

logger = logging.getLogger('research_agent.research')

class ContentFetcher:
    """Async content fetcher with rate limiting"""

    def __init__(self, config):
        self.config = config.research
        self.semaphore = asyncio.Semaphore(self.config.max_concurrent_fetches)
        self.last_request_time = 0.0

    async def fetch_url(self, url: str) -> str:
        """Fetch content from URL with rate limiting"""

        await self.semaphore.acquire()
        try:
            await self._rate_limit_wait()

            logger.info(f"Fetching: {url}")

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url,
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as response:
                    if response.status == 200:
                        content = await response.text()
                        logger.info(f"Fetched {len(content)} chars from {url}")
                        return content
                    else:
                        raise FetchError(f"HTTP {response.status}: {url}")

        finally:
            self.semaphore.release()

    async def fetch_file(self, file_path: str) -> str:
        """Fetch content from local file (WSL path)"""

        logger.info(f"Reading file: {file_path}")

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            logger.info(f"Read {len(content)} chars from {file_path}")
            return content

        except Exception as e:
            raise FetchError(f"Failed to read file {file_path}: {e}")

    async def _rate_limit_wait(self):
        """Enforce rate limiting"""

        current_time = asyncio.get_event_loop().time()
        time_since_last = current_time - self.last_request_time

        min_interval = 1.0 / self.config.rate_limit

        if time_since_last < min_interval:
            wait_time = min_interval - time_since_last
            logger.debug(f"Rate limiting: sleeping {wait_time:.2f}s")
            await asyncio.sleep(wait_time)

        self.last_request_time = asyncio.get_event_loop().time()

    @with_retry(max_retries=3)
    async def fetch_batch(self, urls: List[str]) -> Dict[str, str]:
        """Fetch multiple URLs concurrently"""

        logger.info(f"Fetching batch of {len(urls)} URLs")

        tasks = [self.fetch_url(url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Collect successful results
        successful = {}
        for url, result in zip(urls, results):
            if isinstance(result, Exception):
                logger.error(f"Failed to fetch {url}: {result}")
            else:
                successful[url] = result

        logger.info(f"Successfully fetched {len(successful)}/{len(urls)} URLs")
        return successful

class FetchError(Exception):
    """Content fetch error"""
    pass
```

**Success Criteria**:
- [ ] HTTP fetching works
- [ ] WSL file reading works
- [ ] Rate limiting enforced
- [ ] Concurrent fetching works

**Verification**: Test fetching multiple URLs concurrently

---

### Task 2.2: Content Extractor
**Priority**: P0 | **Effort**: 3 hours | **Dependencies**: None

**File**: `research_agent/research/content_extractor.py`

**Implementation**:
```python
from bs4 import BeautifulSoup
import re
import logging
from typing import Optional

logger = logging.getLogger('research_agent.research')

class ContentExtractor:
    """Extract clean text content from various sources"""

    def extract_text(self, content: str, content_type: str = 'auto') -> str:
        """Extract text content based on content type"""

        if content_type == 'auto':
            content_type = self._detect_content_type(content)

        logger.info(f"Extracting text from {content_type}")

        if content_type == 'html':
            return self._extract_from_html(content)
        elif content_type == 'markdown':
            return self._extract_from_markdown(content)
        else:
            return self._extract_from_plain_text(content)

    def _detect_content_type(self, content: str) -> str:
        """Auto-detect content type"""

        if content.strip().startswith('<'):
            return 'html'
        elif '```' in content or '**' in content:
            return 'markdown'
        else:
            return 'plain'

    def _extract_from_html(self, html: str) -> str:
        """Extract clean text from HTML"""

        soup = BeautifulSoup(html, 'html.parser')

        # Remove script/style/nav/footer/header
        for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside']):
            element.decompose()

        # Extract text
        text = soup.get_text(separator='\n')

        # Clean up excessive newlines
        text = re.sub(r'\n{3,}', '\n\n', text)

        # Remove excessive whitespace
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n[ \t]+\n', '\n', text)

        return text.strip()

    def _extract_from_markdown(self, markdown: str) -> str:
        """Extract text from markdown (simplified)"""

        # Remove code blocks
        text = re.sub(r'```.*?```', '', markdown, flags=re.DOTALL)

        # Remove markdown syntax
        text = re.sub(r'[*_`#]+', '', text)

        # Clean up newlines
        text = re.sub(r'\n{3,}', '\n\n', text)

        return text.strip()

    def _extract_from_plain_text(self, text: str) -> str:
        """Extract and clean plain text"""

        # Remove excessive newlines
        text = re.sub(r'\n{3,}', '\n\n', text)

        # Remove excessive whitespace
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n[ \t]+\n', '\n', text)

        return text.strip()

    def truncate_to_max_length(self, text: str, max_length: int) -> str:
        """Truncate text to maximum length"""

        if len(text) <= max_length:
            return text

        logger.warning(f"Truncating content from {len(text)} to {max_length} chars")
        return text[:max_length]
```

**Success Criteria**:
- [ ] HTML extraction works
- [ ] Markdown extraction works
- [ ] Plain text cleaning works
- [ ] Auto-detection works

**Verification**: Test with sample HTML, markdown, and plain text

---

[CONTINUED IN NEXT DOCUMENT...]
```

**Status**: Tasks 1.1-2.7, 2.1-2.2 detailed above. Remaining tasks in next file.

**Next**: Create Phase 2-5 breakdown document
