-- ================================================================
-- COMPLETE POSTGRESQL SCHEMA FOR HYBRID RAG KNOWLEDGE GRAPH
-- ================================================================
-- Compatible with PostgreSQL 14+
-- Requires: pgvector, pg_trgm, uuid-ossp extensions
-- ================================================================

-- Drop existing tables if recreating
DROP TABLE IF EXISTS edges CASCADE;
DROP TABLE IF EXISTS nodes CASCADE;
DROP TABLE IF EXISTS document_chunks CASCADE;

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS btree_gin;

-- ================================================================
-- NODES TABLE - Entities in the knowledge graph
-- ================================================================
CREATE TABLE nodes (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name TEXT NOT NULL,
    type TEXT NOT NULL CHECK (type IN ('person', 'project', 'organization', 'document', 'concept', 'event', 'location')),
    content TEXT,
    summary TEXT, -- Short summary for quick reference
    metadata JSONB DEFAULT '{}',
    embedding vector(768), -- Google text-embedding-004 dimension
    source_document TEXT, -- Original document reference
    confidence_score FLOAT DEFAULT 1.0 CHECK (confidence_score BETWEEN 0 AND 1),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    created_by TEXT DEFAULT 'system',
    is_active BOOLEAN DEFAULT TRUE,
	-- FIX: Ensure name + type combination is unique
    CONSTRAINT uq_nodes_name_type UNIQUE (name, type)
);
-- Tip: To handle case sensitivity (so "Project Alpha" and "project alpha" don't become duplicates), you can use a functional index instead:
CREATE UNIQUE INDEX idx_nodes_unique_name_type 
ON nodes (LOWER(name), type);
-- Add comments for documentation
COMMENT ON TABLE nodes IS 'Entity nodes in the knowledge graph';
COMMENT ON COLUMN nodes.embedding IS 'Vector embedding (768-dim for Google text-embedding-004)';
COMMENT ON COLUMN nodes.metadata IS 'Flexible JSONB storage for entity-specific attributes';
COMMENT ON COLUMN nodes.confidence_score IS 'Confidence in entity extraction (0-1)';

-- ================================================================
-- EDGES TABLE - Relationships between nodes
-- ================================================================
CREATE TABLE edges (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    source_id UUID NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
    target_id UUID NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
    relationship_type TEXT NOT NULL,
    metadata JSONB DEFAULT '{}',
    weight FLOAT DEFAULT 1.0 CHECK (weight >= 0),
    confidence_score FLOAT DEFAULT 1.0 CHECK (confidence_score BETWEEN 0 AND 1),
    bidirectional BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    created_by TEXT DEFAULT 'system',
    is_active BOOLEAN DEFAULT TRUE,
    CONSTRAINT unique_edge UNIQUE (source_id, target_id, relationship_type),
    CONSTRAINT no_self_reference CHECK (source_id != target_id)
);

-- Add comments
COMMENT ON TABLE edges IS 'Directed relationships between nodes';
COMMENT ON COLUMN edges.weight IS 'Relationship strength/importance';
COMMENT ON COLUMN edges.bidirectional IS 'Whether relationship is symmetric';

-- ================================================================
-- DOCUMENT_CHUNKS TABLE - For RAG chunking strategy
-- ================================================================
CREATE TABLE document_chunks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    metadata JSONB DEFAULT '{}',
    embedding vector(768),
    node_id UUID REFERENCES nodes(id) ON DELETE SET NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    CONSTRAINT unique_chunk UNIQUE (document_id, chunk_index)
);

COMMENT ON TABLE document_chunks IS 'Document chunks for retrieval-augmented generation';

-- ================================================================
-- PERFORMANCE INDEXES
-- ================================================================

-- Node indexes
CREATE INDEX idx_nodes_type ON nodes(type) WHERE is_active = TRUE;
CREATE INDEX idx_nodes_active ON nodes(is_active);
CREATE INDEX idx_nodes_created_at ON nodes(created_at DESC);

-- Trigram indexes for fuzzy text search
CREATE INDEX idx_nodes_name_trgm ON nodes USING GIN (name gin_trgm_ops);
CREATE INDEX idx_nodes_content_trgm ON nodes USING GIN (content gin_trgm_ops);

-- Full-text search indexes
CREATE INDEX idx_nodes_name_fts ON nodes USING GIN (to_tsvector('english', name));
CREATE INDEX idx_nodes_content_fts ON nodes USING GIN (to_tsvector('english', COALESCE(content, '')));

-- Combined text search index for better performance
CREATE INDEX idx_nodes_combined_fts ON nodes 
USING GIN (to_tsvector('english', COALESCE(name, '') || ' ' || COALESCE(content, '') || ' ' || COALESCE(summary, '')));

-- JSONB metadata indexes
CREATE INDEX idx_nodes_metadata ON nodes USING GIN (metadata);
CREATE INDEX idx_edges_metadata ON edges USING GIN (metadata);

-- HNSW index for vector similarity (optimal for most workloads)
-- m=16: number of connections per layer (higher = better recall, more memory)
-- ef_construction=64: search quality during index build (higher = better quality, slower build)
CREATE INDEX idx_nodes_embedding_hnsw ON nodes 
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64)
WHERE embedding IS NOT NULL;

-- Alternative: IVFFlat index (faster build, good for very large datasets)
-- Uncomment if you prefer IVFFlat over HNSW
-- CREATE INDEX idx_nodes_embedding_ivfflat ON nodes 
-- USING ivfflat (embedding vector_cosine_ops)
-- WITH (lists = 100)
-- WHERE embedding IS NOT NULL;

-- Graph traversal indexes
CREATE INDEX idx_edges_source ON edges(source_id) WHERE is_active = TRUE;
CREATE INDEX idx_edges_target ON edges(target_id) WHERE is_active = TRUE;
CREATE INDEX idx_edges_type ON edges(relationship_type) WHERE is_active = TRUE;
CREATE INDEX idx_edges_composite ON edges(source_id, relationship_type) WHERE is_active = TRUE;
CREATE INDEX idx_edges_bidirectional ON edges(target_id, relationship_type) WHERE is_active = TRUE AND bidirectional = TRUE;
CREATE INDEX idx_edges_weight ON edges(weight DESC) WHERE is_active = TRUE;

-- Document chunk indexes
CREATE INDEX idx_chunks_document ON document_chunks(document_id);
CREATE INDEX idx_chunks_node ON document_chunks(node_id) WHERE node_id IS NOT NULL;
CREATE INDEX idx_chunks_embedding_hnsw ON document_chunks 
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64)
WHERE embedding IS NOT NULL;

-- ================================================================
-- TRIGGERS & FUNCTIONS
-- ================================================================

-- Update timestamp trigger
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_nodes_updated_at 
    BEFORE UPDATE ON nodes
    FOR EACH ROW 
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_edges_updated_at 
    BEFORE UPDATE ON edges
    FOR EACH ROW 
    EXECUTE FUNCTION update_updated_at_column();

-- Auto-create bidirectional edges
CREATE OR REPLACE FUNCTION create_bidirectional_edge()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.bidirectional = TRUE THEN
        INSERT INTO edges (source_id, target_id, relationship_type, weight, bidirectional, metadata, created_by)
        VALUES (NEW.target_id, NEW.source_id, NEW.relationship_type, NEW.weight, TRUE, NEW.metadata, NEW.created_by)
        ON CONFLICT (source_id, target_id, relationship_type) DO NOTHING;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER auto_bidirectional_edge
    AFTER INSERT ON edges
    FOR EACH ROW
    WHEN (NEW.bidirectional = TRUE)
    EXECUTE FUNCTION create_bidirectional_edge();

-- ================================================================
-- UTILITY FUNCTIONS
-- ================================================================

-- Function to calculate node importance (PageRank-style)
CREATE OR REPLACE FUNCTION calculate_node_importance(
    node_uuid UUID
)
RETURNS TABLE (
    importance_score FLOAT,
    in_degree BIGINT,
    out_degree BIGINT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        CASE 
            WHEN out_deg > 0 THEN in_deg::FLOAT / out_deg::FLOAT
            ELSE in_deg::FLOAT
        END as importance_score,
        in_deg as in_degree,
        out_deg as out_degree
    FROM (
        SELECT 
            COUNT(DISTINCT e_in.source_id) as in_deg,
            COUNT(DISTINCT e_out.target_id) as out_deg
        FROM nodes n
        LEFT JOIN edges e_in ON n.id = e_in.target_id AND e_in.is_active = TRUE
        LEFT JOIN edges e_out ON n.id = e_out.source_id AND e_out.is_active = TRUE
        WHERE n.id = node_uuid
    ) stats;
END;
$$ LANGUAGE plpgsql;

-- Function to find shortest path between nodes
CREATE OR REPLACE FUNCTION find_shortest_path(
    start_node_id UUID,
    end_node_id UUID,
    max_depth INTEGER DEFAULT 5
)
RETURNS TABLE (
    path_length INTEGER,
    node_path UUID[],
    name_path TEXT[],
    relationship_path TEXT[]
) AS $$
BEGIN
    RETURN QUERY
    WITH RECURSIVE path_search AS (
        -- Base case
        SELECT 
            e.source_id,
            e.target_id,
            e.relationship_type,
            ARRAY[e.source_id, e.target_id] as nodes,
            ARRAY[sn.name, tn.name] as names,
            ARRAY[e.relationship_type] as rels,
            1 as depth
        FROM edges e
        JOIN nodes sn ON e.source_id = sn.id
        JOIN nodes tn ON e.target_id = tn.id
        WHERE e.source_id = start_node_id
          AND e.is_active = TRUE
          AND sn.is_active = TRUE
          AND tn.is_active = TRUE
        
        UNION ALL
        
        -- Recursive case
        SELECT 
            p.source_id,
            e.target_id,
            e.relationship_type,
            p.nodes || e.target_id,
            p.names || tn.name,
            p.rels || e.relationship_type,
            p.depth + 1
        FROM path_search p
        JOIN edges e ON p.target_id = e.source_id
        JOIN nodes tn ON e.target_id = tn.id
        WHERE p.depth < max_depth
          AND NOT e.target_id = ANY(p.nodes)
          AND e.is_active = TRUE
          AND tn.is_active = TRUE
    )
    SELECT 
        depth as path_length,
        nodes as node_path,
        names as name_path,
        rels as relationship_path
    FROM path_search
    WHERE target_id = end_node_id
    ORDER BY depth
    LIMIT 1;
END;
$$ LANGUAGE plpgsql;

-- ================================================================
-- VIEWS FOR COMMON QUERIES
-- ================================================================

-- View: Active knowledge graph summary
CREATE OR REPLACE VIEW vw_graph_summary AS
SELECT 
    (SELECT COUNT(*) FROM nodes WHERE is_active = TRUE) as total_nodes,
    (SELECT COUNT(*) FROM edges WHERE is_active = TRUE) as total_edges,
    (SELECT COUNT(DISTINCT type) FROM nodes WHERE is_active = TRUE) as node_types,
    (SELECT COUNT(DISTINCT relationship_type) FROM edges WHERE is_active = TRUE) as relationship_types,
    (SELECT COUNT(*) FROM nodes WHERE embedding IS NOT NULL AND is_active = TRUE) as nodes_with_embeddings;

-- View: Node degree statistics
CREATE OR REPLACE VIEW vw_node_degrees AS
SELECT 
    n.id,
    n.name,
    n.type,
    COUNT(DISTINCT e_out.id) as out_degree,
    COUNT(DISTINCT e_in.id) as in_degree,
    COUNT(DISTINCT e_out.id) + COUNT(DISTINCT e_in.id) as total_degree
FROM nodes n
LEFT JOIN edges e_out ON n.id = e_out.source_id AND e_out.is_active = TRUE
LEFT JOIN edges e_in ON n.id = e_in.target_id AND e_in.is_active = TRUE
WHERE n.is_active = TRUE
GROUP BY n.id, n.name, n.type;

-- View: Most connected entities
CREATE OR REPLACE VIEW vw_hub_nodes AS
SELECT 
    n.id,
    n.name,
    n.type,
    n.summary,
    d.total_degree,
    d.in_degree,
    d.out_degree,
    ROUND((d.in_degree::FLOAT / NULLIF(d.out_degree, 0))::NUMERIC, 2) as importance_ratio
FROM nodes n
JOIN vw_node_degrees d ON n.id = d.id
WHERE d.total_degree > 0
ORDER BY d.total_degree DESC;


-- ================================================================
-- MAINTENANCE & MONITORING
-- ================================================================

-- Check index usage
CREATE OR REPLACE VIEW vw_index_usage AS
SELECT 
    schemaname,
    relname        AS tablename,
    indexrelname   AS indexname,
    idx_scan,
    idx_tup_read,
    idx_tup_fetch
FROM pg_stat_user_indexes
WHERE schemaname = 'public'
ORDER BY idx_scan DESC;

-- Analyze tables for query optimization
ANALYZE nodes;
ANALYZE edges;
ANALYZE document_chunks;

-- Grant permissions (adjust as needed)
-- GRANT SELECT, INSERT, UPDATE, DELETE ON nodes TO your_app_user;
-- GRANT SELECT, INSERT, UPDATE, DELETE ON edges TO your_app_user;
-- GRANT SELECT, INSERT, UPDATE, DELETE ON document_chunks TO your_app_user;
-- GRANT USAGE ON ALL SEQUENCES IN SCHEMA public TO your_app_user;

-- ================================================================
-- END OF SCHEMA
-- ================================================================
-- Next steps:
-- 1. Run this script: psql -U postgres -d knowledge_graph -f schema.sql
-- 2. Verify extensions: SELECT * FROM pg_extension;
-- 3. Check tables: \dt
-- 4. View indexes: \di
-- ================================================================