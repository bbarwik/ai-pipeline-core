"""ClickHouse DDL statements for the unified database schema.

Three tables:
- execution_nodes: The entire execution DAG
- documents: Document metadata registry
- document_blobs: Content-addressed binary storage
"""

__all__ = [
    "DDL_DOCUMENTS",
    "DDL_DOCUMENT_BLOBS",
    "DDL_EXECUTION_LOGS",
    "DDL_EXECUTION_NODES",
    "DOCUMENTS_TABLE",
    "DOCUMENT_BLOBS_TABLE",
    "EXECUTION_LOGS_TABLE",
    "EXECUTION_NODES_TABLE",
]

EXECUTION_NODES_TABLE = "execution_nodes"
EXECUTION_LOGS_TABLE = "execution_logs"
DOCUMENTS_TABLE = "documents"
DOCUMENT_BLOBS_TABLE = "document_blobs"

DDL_EXECUTION_NODES = f"""
CREATE TABLE IF NOT EXISTS {EXECUTION_NODES_TABLE} (
    -- Identity
    node_id UUID,
    node_kind LowCardinality(String),

    -- Hierarchy
    deployment_id UUID,
    parent_node_id UUID DEFAULT '00000000-0000-0000-0000-000000000000',
    root_deployment_id UUID,

    -- Run context (denormalized)
    run_id String,
    run_scope String,
    deployment_name String,

    -- Ordering
    name String,
    sequence_no Int32,
    attempt Int32 DEFAULT 0,

    -- Class names
    flow_class LowCardinality(String) DEFAULT '',
    task_class LowCardinality(String) DEFAULT '',

    -- Lifecycle
    status LowCardinality(String) DEFAULT 'running',
    started_at DateTime64(3, 'UTC'),
    ended_at Nullable(DateTime64(3, 'UTC')),
    updated_at DateTime64(3, 'UTC'),
    version UInt64 DEFAULT 1,

    -- LLM metrics (conversation_turn nodes)
    model LowCardinality(String) DEFAULT '',
    cost_usd Float64 DEFAULT 0.0,
    tokens_input UInt32 DEFAULT 0,
    tokens_output UInt32 DEFAULT 0,
    tokens_cache_read UInt32 DEFAULT 0,
    tokens_cache_write UInt32 DEFAULT 0,
    tokens_reasoning UInt32 DEFAULT 0,
    turn_count UInt16 DEFAULT 0,

    -- Error
    error_type LowCardinality(String) DEFAULT '',
    error_message String DEFAULT '',

    -- Cross-deployment linking
    remote_child_deployment_id Nullable(UUID),
    parent_deployment_task_id Nullable(UUID),

    -- Cache/resume
    cache_key String DEFAULT '',
    input_fingerprint String DEFAULT '',

    -- Document references (SHA256 arrays)
    input_document_shas Array(String),
    output_document_shas Array(String),
    context_document_shas Array(String),

    -- Denormalized IDs for zero-JOIN filtering
    flow_id Nullable(UUID),
    task_id Nullable(UUID),
    conversation_id Nullable(UUID),

    -- Payload (node-kind-specific JSON data)
    payload String CODEC(ZSTD(3)),

    INDEX idx_node_id node_id TYPE bloom_filter GRANULARITY 1,
    INDEX idx_run_id run_id TYPE bloom_filter GRANULARITY 1,
    INDEX idx_run_scope run_scope TYPE set(0) GRANULARITY 1,
    INDEX idx_cache_key cache_key TYPE bloom_filter GRANULARITY 1,
    INDEX idx_status status TYPE set(8) GRANULARITY 1
)
ENGINE = ReplacingMergeTree(version)
-- Keep the existing ORDER BY key. Our access patterns are still deployment/tree
-- oriented, and the added secondary indexes cover the point and set filters that
-- do not align with this primary key without forcing a broader re-key.
ORDER BY (root_deployment_id, deployment_id, parent_node_id, node_id)
""".strip()

DDL_DOCUMENTS = f"""
CREATE TABLE IF NOT EXISTS {DOCUMENTS_TABLE} (
    -- Identity (content-addressed)
    document_sha256 String,
    content_sha256 String,

    -- Provenance
    deployment_id UUID,
    producing_node_id Nullable(UUID),

    -- Metadata
    document_type LowCardinality(String),
    name String,
    run_scope String DEFAULT '',
    description String DEFAULT '',
    mime_type LowCardinality(String) DEFAULT '',
    size_bytes UInt64 DEFAULT 0,
    publicly_visible Bool DEFAULT false,

    -- Lineage
    derived_from Array(String),
    triggered_by Array(String),

    -- Attachment parallel arrays
    attachment_names Array(String),
    attachment_descriptions Array(String),
    attachment_sha256s Array(String),
    attachment_mime_types Array(String),
    attachment_sizes Array(UInt64),

    -- Mutable fields
    summary String DEFAULT '' CODEC(ZSTD(3)),
    metadata_json String DEFAULT '{{}}' CODEC(ZSTD(3)),

    -- Timestamps
    created_at DateTime64(3, 'UTC'),
    version UInt32 DEFAULT 1,

    INDEX idx_deployment_id deployment_id TYPE bloom_filter GRANULARITY 1,
    INDEX idx_producing_node producing_node_id TYPE bloom_filter GRANULARITY 1,
    INDEX idx_derived_from derived_from TYPE bloom_filter GRANULARITY 1,
    INDEX idx_triggered_by triggered_by TYPE bloom_filter GRANULARITY 1,
    INDEX idx_name name TYPE bloom_filter GRANULARITY 1
)
ENGINE = ReplacingMergeTree(version)
ORDER BY (document_sha256)
""".strip()

DDL_DOCUMENT_BLOBS = f"""
CREATE TABLE IF NOT EXISTS {DOCUMENT_BLOBS_TABLE} (
    -- Identity (content-addressed)
    content_sha256 String,

    -- Content
    content String CODEC(ZSTD(3)),
    size_bytes UInt64 DEFAULT 0,

    -- Timestamp
    created_at DateTime64(3, 'UTC')
)
ENGINE = ReplacingMergeTree()
ORDER BY (content_sha256)
""".strip()

DDL_EXECUTION_LOGS = f"""
CREATE TABLE IF NOT EXISTS {EXECUTION_LOGS_TABLE} (
    node_id UUID,
    deployment_id UUID,
    root_deployment_id UUID,
    flow_id Nullable(UUID),
    task_id Nullable(UUID),
    timestamp DateTime64(3, 'UTC'),
    sequence_no UInt32,
    level LowCardinality(String),
    category LowCardinality(String),
    logger_name String,
    message String,
    event_type LowCardinality(String) DEFAULT '',
    fields String DEFAULT '{{}}',
    exception_text String DEFAULT ''
)
ENGINE = MergeTree()
ORDER BY (root_deployment_id, deployment_id, node_id, sequence_no)
TTL toDateTime(timestamp) + INTERVAL 90 DAY
""".strip()
