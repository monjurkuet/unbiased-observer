-- ================================================================
-- KNOWLEDGE BASE SCHEMA (High-Fidelity GraphRAG)
-- ================================================================
-- Optimized for:
-- 1. Hybrid Search (Vector + Graph)
-- 2. Hierarchical Community Detection (Leiden)
-- 3. Recursive Summarization
-- ================================================================

-- Extensions
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ================================================================
-- 1. CORE GRAPH (Nodes & Edges)
-- ================================================================

CREATE TABLE nodes (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name TEXT NOT NULL,
    type TEXT NOT NULL, -- e.g., 'Person', 'Project', 'Concept'
    description TEXT, -- Comprehensive description from extraction
    
    -- Metadata
    metadata JSONB DEFAULT '{}', -- Store source chunks, confidence, etc.
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    
    -- Vectors
    embedding vector(768), -- For semantic search (Entity Resolution)
    
    -- Constraints
    CONSTRAINT uq_nodes_name_type UNIQUE (name, type)
);

CREATE TABLE edges (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    source_id UUID NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
    target_id UUID NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
    type TEXT NOT NULL, -- e.g., 'AUTHORED', 'LEADS'
    description TEXT, -- Contextual explanation of the relationship
    
    -- Metadata
    weight FLOAT DEFAULT 1.0,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    -- Constraints
    CONSTRAINT uq_edges_source_target_type UNIQUE (source_id, target_id, type)
);

-- ================================================================
-- 2. HIERARCHICAL COMMUNITIES (The "GraphRAG" Layer)
-- ================================================================

CREATE TABLE communities (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    level INTEGER NOT NULL, -- 0=Root/Cluster, 1=Sub-cluster, etc.
    title TEXT NOT NULL, -- Generated title (e.g., "AI Research Cluster")
    summary TEXT, -- The "Community Report"
    full_content TEXT, -- Detailed breakdown
    
    -- Vectors
    embedding vector(768), -- Embedding of the SUMMARY for global search
    
    -- Metadata
    metadata JSONB DEFAULT '{}', -- Store size, period, etc.
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Mapping Nodes to Communities (Many-to-Many)
-- A node can belong to multiple communities at different levels
CREATE TABLE community_membership (
    community_id UUID NOT NULL REFERENCES communities(id) ON DELETE CASCADE,
    node_id UUID NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
    rank FLOAT DEFAULT 0.0, -- Importance of node in this community (PageRank/Degree)
    
    PRIMARY KEY (community_id, node_id)
);

-- Hierarchy (Community -> Parent Community)
CREATE TABLE community_hierarchy (
    child_id UUID NOT NULL REFERENCES communities(id) ON DELETE CASCADE,
    parent_id UUID NOT NULL REFERENCES communities(id) ON DELETE CASCADE,
    
    PRIMARY KEY (child_id, parent_id)
);

-- ================================================================
-- 3. TEMPORAL DATA (Events & Timelines)
-- ================================================================

CREATE TABLE events (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    node_id UUID REFERENCES nodes(id) ON DELETE CASCADE, -- The primary entity involved
    description TEXT NOT NULL, -- e.g., "Announced Project Chimera"
    timestamp TIMESTAMPTZ, -- Normalized date
    raw_time_desc TEXT, -- Original text (e.g., "Q1 2026")
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- ================================================================
-- 4. INDEXES
-- ================================================================

-- Text Search (Trigram)
CREATE INDEX idx_nodes_name_trgm ON nodes USING GIN (name gin_trgm_ops);
CREATE INDEX idx_nodes_desc_fts ON nodes USING GIN (to_tsvector('english', description));
CREATE INDEX idx_communities_title_trgm ON communities USING GIN (title gin_trgm_ops);

-- Vector Search (HNSW)
-- Optimized for "text-embedding-004" (768d)
CREATE INDEX idx_nodes_embedding ON nodes USING hnsw (embedding vector_cosine_ops);
CREATE INDEX idx_communities_embedding ON communities USING hnsw (embedding vector_cosine_ops);

-- Graph Traversal
CREATE INDEX idx_edges_source ON edges(source_id);
CREATE INDEX idx_edges_target ON edges(target_id);
CREATE INDEX idx_community_mem_node ON community_membership(node_id);
CREATE INDEX idx_community_mem_comm ON community_membership(community_id);

-- ================================================================
-- 4. UTILITY FUNCTIONS
-- ================================================================

-- Update timestamp
CREATE OR REPLACE FUNCTION update_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_nodes_modtime
    BEFORE UPDATE ON nodes
    FOR EACH ROW EXECUTE FUNCTION update_timestamp();

CREATE TRIGGER update_communities_modtime
    BEFORE UPDATE ON communities
    FOR EACH ROW EXECUTE FUNCTION update_timestamp();
