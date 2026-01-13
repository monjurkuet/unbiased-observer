-- ================================================================
-- AUTONOMOUS AGENT DATABASE SCHEMA
-- ================================================================
-- Optimized for:
-- 1. Hybrid Search (Vector + Graph)
-- 2. Hierarchical Community Detection (Leiden)
-- 3. Recursive Summarization
-- 4. Task Queue (Persistent, scalable)
-- ================================================================

-- Extensions
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ================================================================
-- 1. CORE GRAPH (Existing - from knowledge_base/schema.sql)
-- ================================================================

-- NOTE: These tables already exist in knowledge_base
-- We'll extend them with additional indexes/views if needed

-- ================================================================
-- 2. HIERARCHICAL COMMUNITIES (Existing - from knowledge_base/schema.sql)
-- ================================================================

-- NOTE: These tables already exist in knowledge_base
-- We'll extend them with additional indexes/views if needed

-- ================================================================
-- 3. TEMPORAL DATA (Existing - from knowledge_base/schema.sql)
-- ================================================================

-- NOTE: These tables already exist in knowledge_base
-- We'll extend them with additional indexes/views if needed

-- ================================================================
-- 4. AGENT TASK QUEUE (NEW)
-- ================================================================

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
CREATE INDEX IF NOT EXISTS idx_tasks_type_status ON research_tasks(type, status);

-- ================================================================
-- 5. AGENT STATE (NEW)
-- ================================================================

CREATE TABLE IF NOT EXISTS agent_state (
    key TEXT PRIMARY KEY,
    value JSONB NOT NULL,
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_state_updated ON agent_state(updated_at);

-- ================================================================
-- 6. RESEARCH SOURCES (NEW)
-- ================================================================

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

-- ================================================================
-- 7. INGESTION LOGS (NEW)
-- ================================================================

CREATE TABLE IF NOT EXISTS ingestion_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    source TEXT NOT NULL,
    task_id UUID REFERENCES research_tasks(id) ON DELETE CASCADE,
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

-- ================================================================
-- 8. PROCESSING LOGS (NEW)
-- ================================================================

CREATE TABLE IF NOT EXISTS processing_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    task_id UUID REFERENCES research_tasks(id) ON DELETE CASCADE,
    status TEXT NOT NULL,
    nodes_processed INTEGER DEFAULT 0,
    communities_created INTEGER DEFAULT 0,
    duration_seconds FLOAT,
    error_message TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_processing_task ON processing_logs(task_id);
CREATE INDEX IF NOT EXISTS idx_processing_status ON processing_logs(status);
CREATE INDEX IF NOT EXISTS idx_processing_created ON processing_logs(created_at);

-- ================================================================
-- TRIGGERS FOR AUTOMATIC TIMESTAMP UPDATES
-- ================================================================

-- Trigger: Update timestamps on task state changes
CREATE TRIGGER update_tasks_modtime
    BEFORE UPDATE ON research_tasks
    FOR EACH ROW
    EXECUTE FUNCTION update_timestamp()
;

-- Trigger: Update timestamps on agent state changes
CREATE TRIGGER update_state_modtime
    BEFORE UPDATE ON agent_state
    FOR EACH ROW
    EXECUTE FUNCTION update_timestamp();

-- Trigger: Update last_fetched_at on research sources
CREATE TRIGGER update_sources_timestamp
    BEFORE UPDATE ON research_sources
    FOR EACH ROW
    WHEN (NEW.last_fetched_at IS DISTINCT FROM OLD.last_fetched_at)
    EXECUTE FUNCTION update_timestamp();

-- ================================================================
-- VIEWS FOR QUERIES (NEW)
-- ================================================================

-- View: Pending tasks by priority
CREATE OR REPLACE VIEW v_pending_tasks AS
SELECT
    id,
    type,
    source,
    created_at,
    ROW_NUMBER() OVER (PARTITION BY type ORDER BY created_at ASC) AS task_order
FROM research_tasks
WHERE status = 'PENDING'
ORDER BY task_order;

-- View: Failed tasks by retry count
CREATE OR REPLACE VIEW v_failed_tasks AS
SELECT
    id,
    type,
    retry_count,
    error_message,
    created_at
FROM research_tasks
WHERE status = 'FAILED'
ORDER BY retry_count DESC, created_at DESC;

-- View: Queue status summary
CREATE OR REPLACE VIEW v_queue_status AS
SELECT
    status,
    COUNT(*) as count
FROM research_tasks
GROUP BY status;

-- View: Recent activity
CREATE OR REPLACE VIEW v_recent_activity AS
SELECT
    type,
    COUNT(*) as count,
    MAX(created_at) as latest_activity
FROM research_tasks
WHERE created_at > NOW() - INTERVAL '1 hour'
GROUP BY type
ORDER BY latest_activity DESC;

-- ================================================================
-- FUNCTIONS FOR COMMON OPERATIONS
-- ================================================================

-- Function: Update task with status
CREATE OR REPLACE FUNCTION update_task_status(task_id UUID, new_status TEXT, error_message TEXT DEFAULT NULL, error_details JSONB DEFAULT NULL)
RETURNS VOID AS $$
BEGIN
    UPDATE research_tasks
    SET status = new_status,
        completed_at = CASE WHEN new_status IN ('COMPLETED', 'FAILED') THEN NOW() ELSE NULL END,
        error_message = update_task_status.error_message,
        error_details = COALESCE(update_task_status.error_details, '{}')
    WHERE id = task_id;
END;
$$ LANGUAGE plpgsql;

-- Function: Mark task as failed with error details
CREATE OR REPLACE FUNCTION mark_task_failed(task_id UUID, error_message TEXT, error_details JSONB DEFAULT NULL)
RETURNS VOID AS $$
BEGIN
    UPDATE research_tasks
    SET status = 'FAILED',
        completed_at = NOW(),
        error_message = mark_task_failed.error_message,
        error_details = mark_task_failed.error_details,
        retry_count = retry_count + 1
    WHERE id = task_id;
END;
$$ LANGUAGE plpgsql;

-- Function: Get next task
CREATE OR REPLACE FUNCTION get_next_task(worker_id TEXT, task_type TEXT DEFAULT NULL)
RETURNS TABLE(id, type, source, metadata, status) AS $$
DECLARE
    task_id research_tasks.id%TYPE;
BEGIN
    UPDATE research_tasks
    SET status = 'IN_PROGRESS',
        worker_id = get_next_task.worker_id,
        started_at = NOW()
    WHERE id = (
        SELECT id FROM research_tasks
        WHERE status = 'PENDING'
        AND (task_type IS NULL OR type = task_type)
        ORDER BY created_at ASC
        LIMIT 1
        FOR UPDATE SKIP LOCKED
    )
    RETURNING id, type, source, metadata, status;
END;
$$ LANGUAGE plpgsql;

-- Function: Increment retry count
CREATE OR REPLACE FUNCTION increment_task_retry(task_id UUID)
RETURNS VOID AS $$
BEGIN
    UPDATE research_tasks
    SET retry_count = retry_count + 1,
        status = CASE WHEN retry_count + 1 >= max_retries THEN 'FAILED' ELSE 'PENDING' END,
        error_message = NULL,
        error_details = '{}'
    WHERE id = task_id;
END;
$$ LANGUAGE plpgsql;

-- Function: Get failed tasks for retry
CREATE OR REPLACE FUNCTION get_failed_tasks(retry_limit INTEGER DEFAULT 3)
RETURNS TABLE(id, type, source, metadata, status, retry_count, error_message) AS $$
BEGIN
    RETURN QUERY
    SELECT
        id, type, source, metadata, retry_count, error_message
    FROM research_tasks
    WHERE status = 'FAILED'
    AND retry_count < retry_limit
    ORDER BY created_at ASC
    LIMIT 100;
END;
$$ LANGUAGE plpgsql;

-- Function: Get pending count
CREATE OR REPLACE FUNCTION get_pending_count(task_type TEXT DEFAULT NULL)
RETURNS INTEGER AS $$
BEGIN
    SELECT COUNT(*)
    FROM research_tasks
    WHERE status = 'PENDING'
    AND (task_type IS NULL OR type = task_type);
END;
$$ LANGUAGE plpgsql;

-- Function: Update timestamp
CREATE OR REPLACE FUNCTION update_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- ================================================================
-- GRANTS (NEW)
-- ================================================================

-- Grant agentzero all necessary permissions
-- NOTE: This should be done manually by database admin
-- Example:
-- GRANT ALL PRIVILEGES ON DATABASE knowledge_graph TO agentzero;
-- GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO agentzero;

-- Grant usage permission
GRANT USAGE ON SCHEMA public TO agentzero;
