# Comprehensive Prefect v3.4.13 Tutorial for AI Agents

## Overview: Modern workflow orchestration with Prefect v3

Prefect v3.4.13 represents a major advancement in workflow orchestration, offering **90% performance improvements** over previous versions, open-sourced events and automations, and a powerful transactional system for building resilient data pipelines. This tutorial provides everything needed to implement production-grade logging and workflow orchestration using Prefect's latest features.

## 1. Flow and Task Decorators with Complete Parameters

### The @flow Decorator

The flow decorator transforms Python functions into observable, orchestrated workflows with comprehensive configuration options:

```python
from prefect import flow
from datetime import datetime

@flow(
    name="data-processing-flow",                      # Optional custom name
    version="1.0.0",                                 # Version tracking
    description="Processes daily data batches",       # Documentation
    flow_run_name="processing-{date:%Y-%m-%d}",      # Dynamic run naming
    timeout_seconds=3600,                            # Maximum runtime
    validate_parameters=True,                        # Pydantic validation
    retries=3,                                       # Retry attempts
    retry_delay_seconds=60,                          # Delay between retries
    persist_result=True,                             # Store results
    log_prints=True,                                 # Capture print statements
    on_failure=[failure_handler],                    # Failure hooks
    on_completion=[success_handler]                  # Success hooks
)
def data_processing_flow(
    input_file: str,
    batch_size: int = 1000,
    date: datetime = None
):
    if date is None:
        date = datetime.now()
    print(f"Processing {input_file} with batch size {batch_size}")
    return {"status": "completed", "records": batch_size}
```

### The @task Decorator

Tasks are the building blocks of flows, representing discrete units of work:

```python
from prefect import task
from prefect.tasks import task_input_hash
from datetime import timedelta

@task(
    name="data-transform-task",                      # Custom task name
    description="Transforms raw data",               # Documentation
    tags={"data", "transform", "critical"},         # Categorization tags
    retries=3,                                       # Retry configuration
    retry_delay_seconds=[1, 5, 10],                 # Progressive delays
    retry_jitter_factor=0.1,                        # Avoid thundering herd
    timeout_seconds=300,                            # Task timeout
    log_prints=True,                                # Log print statements
    cache_key_fn=task_input_hash,                   # Cache strategy
    cache_expiration=timedelta(hours=1),            # Cache duration
    persist_result=True,                            # Store results
    on_failure=[task_failure_handler],              # Failure handling
    on_completion=[task_success_handler]            # Success handling
)
def transform_data(raw_data: list, transformation_type: str) -> dict:
    print(f"Transforming {len(raw_data)} records using {transformation_type}")
    # Transformation logic here
    return {"transformed": len(raw_data), "type": transformation_type}
```

### Type Validation with Pydantic

Prefect v3 uses Pydantic v2 for robust parameter validation:

```python
from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum

class ProcessingMode(str, Enum):
    FAST = "fast"
    THOROUGH = "thorough"
    BALANCED = "balanced"

class DataConfig(BaseModel):
    input_path: str = Field(..., description="Input data location")
    output_path: str = Field(..., description="Output destination")
    chunk_size: int = Field(default=1000, gt=0, le=10000)
    processing_mode: ProcessingMode = ProcessingMode.BALANCED
    tags: Optional[List[str]] = None

@flow(validate_parameters=True)
def validated_flow(config: DataConfig):
    """Flow with automatic parameter validation and type coercion."""
    print(f"Processing from {config.input_path} to {config.output_path}")
    print(f"Mode: {config.processing_mode.value}, Chunk: {config.chunk_size}")
```

## 2. Flow Execution Patterns

### Synchronous vs Asynchronous Flows

```python
from prefect import flow, task
import asyncio
import time

# Synchronous flow - blocks until complete
@task
def sync_task(duration: int) -> str:
    time.sleep(duration)
    return f"Completed after {duration}s"

@flow
def sync_flow():
    result1 = sync_task(2)
    result2 = sync_task(3)
    return [result1, result2]  # Sequential execution

# Asynchronous flow - enables concurrent operations
@task
async def async_task(duration: int) -> str:
    await asyncio.sleep(duration)
    return f"Async completed after {duration}s"

@flow
async def async_flow():
    # Concurrent execution with gather
    results = await asyncio.gather(
        async_task(2),
        async_task(3),
        async_task(1)
    )
    return results

# Run async flow
results = asyncio.run(async_flow())
```

### Parallel Execution with Task Runners

```python
from prefect import flow, task
from prefect.task_runners import ThreadPoolTaskRunner, DaskTaskRunner

@task
def process_item(item: int) -> int:
    # CPU or I/O intensive work
    return item ** 2

# ThreadPool for I/O-bound tasks
@flow(task_runner=ThreadPoolTaskRunner(max_workers=5))
def parallel_io_flow(items: list):
    futures = [process_item.submit(item) for item in items]
    results = [f.result() for f in futures]
    return results

# Dask for CPU-intensive distributed processing
@flow(task_runner=DaskTaskRunner(
    cluster_kwargs={"n_workers": 4, "threads_per_worker": 2}
))
def distributed_cpu_flow(items: list):
    futures = process_item.map(items)
    return futures.result()
```

## 3. Task Submission Methods

### Direct Execution vs .submit() vs .map()

```python
from prefect import flow, task
from prefect.futures import wait

@task
def process_data(data: str) -> str:
    return f"processed_{data}"

@flow
def submission_patterns_flow():
    # Direct execution - synchronous, blocking
    direct_result = process_data("direct")

    # .submit() - returns future immediately
    future1 = process_data.submit("async1")
    future2 = process_data.submit("async2")

    # Wait for futures
    wait([future1, future2])
    submit_results = [future1.result(), future2.result()]

    # .map() - parallel processing over iterables
    items = ["item1", "item2", "item3", "item4"]
    map_futures = process_data.map(items)
    map_results = map_futures.result()

    return {
        "direct": direct_result,
        "submit": submit_results,
        "map": map_results
    }
```

### Working with return_state Parameter

```python
from prefect import flow, task
from prefect.states import Failed

@task
def risky_task(value: int):
    if value < 0:
        raise ValueError("Negative value not allowed")
    return value * 2

@flow
def state_handling_flow():
    # Get state instead of result
    future = risky_task.submit(-5)
    state = future.wait()

    if state.is_failed():
        print(f"Task failed: {state.message}")
        # Implement recovery logic
        recovery_future = risky_task.submit(5)
        return recovery_future.result()

    return state.result()
```

## 4. State Management and Error Handling

### Comprehensive State System

```python
from prefect import flow, task
from prefect.states import Completed, Failed

@task(retries=3, retry_delay_seconds=[1, 5, 10])
def resilient_task(value: int):
    """Task with automatic retry on failure."""
    if value < 0:
        raise ValueError("Temporary failure")
    return value * 2

def handle_task_failure(task, task_run, state):
    """Custom failure handler."""
    print(f"Task {task.name} failed in run {task_run.id}")
    # Send alerts, clean up resources, etc.

@task(on_failure=[handle_task_failure])
def monitored_task():
    # Task with failure monitoring
    raise Exception("Monitored failure")

@flow
def error_handling_flow():
    try:
        # Primary operation with retries
        result = resilient_task(10)
    except Exception as e:
        print(f"All retries exhausted: {e}")
        # Return explicit state
        return Failed(f"Flow failed: {e}")

    return Completed(f"Success: {result}")
```

### Transaction Support with Rollback

```python
from prefect import flow, task
from prefect.transactions import transaction
import os

@task
def write_critical_file(content: str, filename: str):
    with open(filename, 'w') as f:
        f.write(content)
    return filename

@write_critical_file.on_rollback
def cleanup_file(transaction):
    """Rollback handler to clean up on failure."""
    filename = transaction.get("filename")
    if filename and os.path.exists(filename):
        os.remove(filename)
        print(f"Rolled back: removed {filename}")

@flow
def transactional_flow():
    with transaction():
        # Write file in transaction
        file1 = write_critical_file("data1", "output1.txt")

        # This failure will trigger rollback of file1
        if not validate_data():
            raise ValueError("Validation failed")

        file2 = write_critical_file("data2", "output2.txt")
        return [file1, file2]
```

## 5. Logging Integration

### Using get_run_logger and log_prints

```python
from prefect import flow, task
from prefect.logging import get_run_logger

@task(log_prints=True)
def logged_task(data: list):
    logger = get_run_logger()

    # Structured logging with different levels
    logger.info(f"Processing {len(data)} items")
    logger.debug(f"Data sample: {data[:3]}")

    # Print statements automatically logged when log_prints=True
    print(f"This print is captured as a log")

    try:
        result = process_data(data)
        logger.info(f"Successfully processed {len(result)} items")
        return result
    except Exception as e:
        logger.error(f"Processing failed: {e}", exc_info=True)
        raise

@flow(log_prints=True)
def logging_flow():
    logger = get_run_logger()
    logger.info("Flow started")

    # Configure logging globally
    import os
    os.environ["PREFECT_LOGGING_LOG_PRINTS"] = "true"
    os.environ["PREFECT_LOGGING_LEVEL"] = "DEBUG"

    data = [1, 2, 3, 4, 5]
    result = logged_task(data)

    logger.info(f"Flow completed with result: {result}")
    return result
```

### Custom Logging Configuration

```python
# Set custom log formatting
import os
os.environ["PREFECT_LOGGING_FORMATTERS_STANDARD_FLOW_RUN_FMT"] = \
    "%(asctime)s | %(levelname)s | %(flow_run_id)s | %(message)s"

# Include external library loggers
os.environ["PREFECT_LOGGING_EXTRA_LOGGERS"] = "boto3,requests,sqlalchemy"

@flow
def custom_logging_flow():
    logger = get_run_logger()

    # Logs include flow run ID and custom format
    logger.info("Custom formatted log message")

    # External library logs also captured
    import requests
    response = requests.get("https://api.example.com/data")

    return response.status_code
```

## 6. Deployment Methods

### Using flow.deploy() for Production

```python
from prefect import flow

@flow(log_prints=True)
def production_flow(environment: str = "prod"):
    print(f"Running in {environment} environment")
    # Production workflow logic
    return {"status": "success", "env": environment}

if __name__ == "__main__":
    # Deploy to work pool with Docker
    production_flow.deploy(
        name="production-deployment",
        work_pool_name="docker-pool",
        image="my-registry/prefect-flows:v1.0.0",

        # Build and push Docker image
        build=True,
        push=True,

        # Schedule configuration
        cron="0 6 * * *",  # Daily at 6 AM
        timezone="America/New_York",

        # Runtime configuration
        parameters={"environment": "production"},
        tags=["production", "daily", "critical"],

        # Infrastructure customization
        job_variables={
            "env": {
                "DATABASE_URL": "{{ prefect.variables.prod_db_url }}",
                "API_KEY": "{{ prefect.blocks.secret.prod-api-key }}"
            },
            "resources": {
                "memory": "4Gi",
                "cpu": "2000m"
            }
        }
    )
```

### Using flow.serve() for Local Development

```python
from prefect import flow, serve

@flow
def development_flow():
    print("Development flow running")
    return "dev_result"

@flow
def testing_flow():
    print("Testing flow running")
    return "test_result"

if __name__ == "__main__":
    # Serve multiple flows locally
    serve(
        development_flow.to_deployment(
            name="dev-deployment",
            interval=60,  # Run every minute
            tags=["development"]
        ),
        testing_flow.to_deployment(
            name="test-deployment",
            cron="*/5 * * * *",  # Every 5 minutes
            tags=["testing"]
        ),
        limit=5  # Maximum concurrent runs
    )
```

## 7. Concurrency and Parallel Execution

### Task Run Concurrency Limits

```python
from prefect import flow, task
import time

# Create concurrency limit via CLI or API
# prefect concurrency-limit create --tag "database" --concurrency-limit 3

@task(tags=["database"])
def database_operation(query_id: int):
    """Task limited by database concurrency tag."""
    time.sleep(2)  # Simulate database query
    return f"Query {query_id} completed"

@flow
def concurrent_database_flow():
    # Only 3 will run simultaneously due to concurrency limit
    futures = [database_operation.submit(i) for i in range(10)]
    results = [f.result() for f in futures]
    return results
```

### Advanced Task Runners

```python
from prefect import flow, task
from prefect.task_runners import DaskTaskRunner, RayTaskRunner

# Dask for distributed CPU-intensive work
@flow(task_runner=DaskTaskRunner(
    cluster_kwargs={
        "n_workers": 4,
        "threads_per_worker": 2,
        "processes": True
    },
    adapt_kwargs={
        "minimum": 1,
        "maximum": 10,
        "target_duration": "30s"  # Auto-scale based on queue
    }
))
def dask_distributed_flow(data: list):
    futures = cpu_intensive_task.map(data)
    return futures.result()

# Ray for ML workloads
@flow(task_runner=RayTaskRunner(
    init_kwargs={
        "num_cpus": 8,
        "num_gpus": 2,
        "dashboard_host": "0.0.0.0"
    }
))
def ml_training_flow(dataset: str):
    preprocessed = preprocess_data.submit(dataset)
    model = train_model.submit(preprocessed)
    metrics = evaluate_model.submit(model)
    return metrics.result()
```

## 8. Retry, Timeout, and Error Handling

### Advanced Retry Strategies

```python
from prefect import flow, task
from prefect.tasks import exponential_backoff
import httpx

def should_retry_request(task, task_run, state) -> bool:
    """Custom retry logic based on error type."""
    try:
        state.result()
    except httpx.HTTPStatusError as e:
        # Don't retry on 4xx errors
        return not (400 <= e.response.status_code < 500)
    except httpx.ConnectError:
        # Always retry connection errors
        return True
    except Exception:
        # Retry other exceptions
        return True

@task(
    retries=5,
    retry_delay_seconds=exponential_backoff(backoff_factor=2),
    retry_jitter_factor=0.5,  # Add randomness to prevent thundering herd
    retry_condition_fn=should_retry_request
)
def resilient_api_call(url: str):
    response = httpx.get(url, timeout=30)
    response.raise_for_status()
    return response.json()

@flow(timeout_seconds=300)  # 5-minute timeout for entire flow
def timeout_aware_flow():
    try:
        # Will retry with exponential backoff: 2s, 4s, 8s, 16s, 32s
        data = resilient_api_call("https://api.example.com/data")
        return data
    except Exception as e:
        print(f"All retries exhausted or timeout reached: {e}")
        # Implement fallback logic
        return {"fallback": "data"}
```

## 9. Production Best Practices

### Structured Flow Organization

```python
from prefect import flow, task
from prefect.artifacts import create_markdown_artifact
from prefect.logging import get_run_logger
from typing import List, Dict
import pandas as pd

# Separate concerns into focused tasks
@task(
    name="extract-data",
    retries=3,
    cache_key_fn=task_input_hash,
    cache_expiration=timedelta(hours=1)
)
def extract_data(source: str) -> pd.DataFrame:
    """Extract with caching and retries."""
    logger = get_run_logger()
    logger.info(f"Extracting from {source}")

    # Extract logic with proper error handling
    try:
        data = pd.read_sql(query, connection)
        logger.info(f"Extracted {len(data)} records")
        return data
    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        raise

@task(name="validate-data")
def validate_data(data: pd.DataFrame) -> pd.DataFrame:
    """Data quality validation."""
    logger = get_run_logger()

    # Quality checks
    if data.empty:
        raise ValueError("No data to process")

    null_count = data.isnull().sum().sum()
    duplicate_count = data.duplicated().sum()

    # Create quality report artifact
    report = f"""
    # Data Quality Report
    - Records: {len(data)}
    - Null values: {null_count}
    - Duplicates: {duplicate_count}
    - Memory: {data.memory_usage().sum() / 1024**2:.2f} MB
    """

    create_markdown_artifact(
        key="data-quality",
        markdown=report,
        description="Data quality metrics"
    )

    if null_count > len(data) * 0.1:  # >10% nulls
        raise ValueError(f"Data quality issue: {null_count} null values")

    return data

@task(name="transform-data")
def transform_data(data: pd.DataFrame, rules: Dict) -> pd.DataFrame:
    """Apply business transformations."""
    logger = get_run_logger()
    logger.info(f"Applying {len(rules)} transformation rules")

    # Apply transformations
    for rule_name, rule_func in rules.items():
        data = rule_func(data)
        logger.debug(f"Applied rule: {rule_name}")

    return data

@task(name="load-data")
def load_data(data: pd.DataFrame, destination: str) -> Dict:
    """Load with transaction support."""
    logger = get_run_logger()

    with transaction():
        # Write data
        data.to_sql(destination, connection, if_exists='replace')
        logger.info(f"Loaded {len(data)} records to {destination}")

        # Return metadata
        return {
            "destination": destination,
            "records_loaded": len(data),
            "timestamp": datetime.now()
        }

@flow(
    name="production-etl",
    description="Production ETL pipeline with best practices",
    log_prints=True,
    retries=1,
    on_failure=[send_alert_on_failure]
)
def production_etl_flow(
    source: str,
    destination: str,
    transformation_rules: Dict
) -> Dict:
    """Production-ready ETL pipeline."""
    logger = get_run_logger()
    logger.info("Starting ETL pipeline")

    # Extract
    raw_data = extract_data(source)

    # Validate
    validated_data = validate_data(raw_data)

    # Transform
    transformed_data = transform_data(validated_data, transformation_rules)

    # Load
    result = load_data(transformed_data, destination)

    logger.info(f"ETL completed: {result}")
    return result
```

### Monitoring and Observability

```python
from prefect import flow, task
from prefect.artifacts import create_link_artifact, create_progress_artifact
from prefect.events import emit_event
import time

@task
def monitored_batch_processing(items: List[str]) -> List[str]:
    """Process with progress tracking."""
    results = []
    total = len(items)

    for i, item in enumerate(items):
        # Update progress artifact
        progress = ((i + 1) / total) * 100
        create_progress_artifact(
            key="batch-progress",
            progress=progress,
            description=f"Processing item {i+1} of {total}"
        )

        # Process item
        result = process_single_item(item)
        results.append(result)

        # Emit custom event every 10 items
        if (i + 1) % 10 == 0:
            emit_event(
                event="batch.milestone",
                resource={"prefect.resource.id": "batch-processor"},
                payload={"processed": i + 1, "total": total}
            )

    # Create completion artifact
    create_link_artifact(
        key="results-location",
        link="s3://bucket/results/batch_output.json",
        description="Batch processing results"
    )

    return results
```

## 10. External System Integration

### Database Integration Pattern

```python
from prefect import flow, task
from prefect.blocks.system import Secret
import sqlalchemy
from contextlib import contextmanager

@contextmanager
def get_db_connection():
    """Secure database connection management."""
    db_password = Secret.load("database-password")

    engine = sqlalchemy.create_engine(
        f"postgresql://user:{db_password.get()}@host/database",
        pool_size=5,
        max_overflow=10,
        pool_pre_ping=True  # Verify connections
    )

    try:
        with engine.connect() as conn:
            yield conn
    finally:
        engine.dispose()

@task(retries=3)
def database_query(query: str) -> pd.DataFrame:
    """Execute database query with connection pooling."""
    with get_db_connection() as conn:
        return pd.read_sql(query, conn)

@flow
def database_integration_flow():
    # Parallel database queries with connection pooling
    futures = [
        database_query.submit("SELECT * FROM table1"),
        database_query.submit("SELECT * FROM table2"),
        database_query.submit("SELECT * FROM table3")
    ]

    results = [f.result() for f in futures]
    return results
```

### Cloud Service Integration

```python
from prefect import flow, task
from prefect_aws import S3Bucket
from prefect_gcp import GcsBucket
import pandas as pd

@task
async def multi_cloud_data_sync(source_bucket: str, dest_bucket: str):
    """Sync data across cloud providers."""
    # Load cloud storage blocks
    s3_block = await S3Bucket.load("aws-data-bucket")
    gcs_block = await GcsBucket.load("gcp-data-bucket")

    # Read from S3
    s3_data = await s3_block.read_path("data/input.parquet")
    df = pd.read_parquet(s3_data)

    # Process data
    processed_df = transform_data(df)

    # Write to GCS
    output_path = f"processed/output_{datetime.now():%Y%m%d}.parquet"
    await gcs_block.write_path(output_path, processed_df.to_parquet())

    return {"records_processed": len(processed_df), "output": output_path}
```

### Event-Driven Integration

```python
from prefect import flow
from prefect.events import DeploymentEventTrigger

@flow
def event_driven_flow(event_data: dict):
    """Flow triggered by external events."""
    print(f"Triggered by event: {event_data}")

    # Process based on event type
    if event_data.get("type") == "file.uploaded":
        process_uploaded_file(event_data["file_path"])
    elif event_data.get("type") == "api.webhook":
        handle_webhook(event_data["payload"])

    return {"processed": True, "event_id": event_data.get("id")}

# Deploy with event trigger
event_driven_flow.deploy(
    name="event-processor",
    work_pool_name="kubernetes-pool",
    triggers=[
        DeploymentEventTrigger(
            enabled=True,
            match={
                "external.file.uploaded": {"bucket": ["data-bucket"]},
                "external.api.webhook": {"source": ["partner-api"]}
            },
            parameters={"event_data": "{{ event }}"}
        )
    ]
)
```

## Complete Production Example: Data Pipeline with All Features

```python
from prefect import flow, task, serve
from prefect.task_runners import DaskTaskRunner
from prefect.logging import get_run_logger
from prefect.artifacts import create_markdown_artifact
from prefect.transactions import transaction
from prefect.tasks import exponential_backoff, task_input_hash
from prefect.blocks.system import Secret
from datetime import datetime, timedelta
from typing import List, Dict
import pandas as pd
import asyncio

# Configuration
class PipelineConfig(BaseModel):
    source_database: str
    destination_bucket: str
    batch_size: int = 10000
    parallel_workers: int = 4
    cache_hours: int = 1

# Tasks with all production features
@task(
    name="extract-source-data",
    retries=3,
    retry_delay_seconds=exponential_backoff(backoff_factor=2),
    cache_key_fn=task_input_hash,
    cache_expiration=timedelta(hours=1),
    tags=["database", "extract"]
)
async def extract_data(config: PipelineConfig, date_range: tuple) -> pd.DataFrame:
    """Extract data with caching, retries, and monitoring."""
    logger = get_run_logger()
    logger.info(f"Extracting data for range: {date_range}")

    # Secure credential handling
    db_password = await Secret.load("db-password")

    # Extract with monitoring
    start_time = datetime.now()
    data = await fetch_from_database(config.source_database, date_range, db_password)
    duration = (datetime.now() - start_time).total_seconds()

    logger.info(f"Extracted {len(data)} records in {duration:.2f}s")
    return data

@task(
    name="validate-transform",
    log_prints=True,
    on_failure=[cleanup_on_failure]
)
def validate_and_transform(data: pd.DataFrame, rules: Dict) -> pd.DataFrame:
    """Validate and transform with quality reporting."""
    logger = get_run_logger()

    # Data quality validation
    quality_metrics = {
        "input_records": len(data),
        "null_values": data.isnull().sum().sum(),
        "duplicates": data.duplicated().sum()
    }

    print(f"Quality metrics: {quality_metrics}")

    # Create quality artifact
    create_markdown_artifact(
        key="data-quality-report",
        markdown=f"""
        # Data Quality Report
        - Input Records: {quality_metrics['input_records']}
        - Null Values: {quality_metrics['null_values']}
        - Duplicates: {quality_metrics['duplicates']}
        """,
        description="Pre-transformation quality metrics"
    )

    # Apply transformations
    transformed = apply_transformation_rules(data, rules)

    logger.info(f"Transformed {len(transformed)} records")
    return transformed

@task(
    name="parallel-enrichment",
    tags=["api", "enrichment"]
)
async def enrich_data_batch(batch: pd.DataFrame, api_endpoint: str) -> pd.DataFrame:
    """Enrich data batch with external API data."""
    logger = get_run_logger()

    enriched_records = []
    for _, record in batch.iterrows():
        enriched = await call_enrichment_api(api_endpoint, record)
        enriched_records.append(enriched)

    logger.info(f"Enriched {len(enriched_records)} records")
    return pd.DataFrame(enriched_records)

@task(name="load-to-destination")
def load_data(data: pd.DataFrame, destination: str) -> Dict:
    """Load data with transaction support."""
    logger = get_run_logger()

    with transaction():
        # Write data with rollback capability
        output_path = f"{destination}/data_{datetime.now():%Y%m%d_%H%M%S}.parquet"
        data.to_parquet(output_path)

        logger.info(f"Loaded {len(data)} records to {output_path}")

        return {
            "path": output_path,
            "records": len(data),
            "timestamp": datetime.now().isoformat()
        }

# Main production flow
@flow(
    name="production-data-pipeline",
    description="Complete production data pipeline with all features",
    task_runner=DaskTaskRunner(
        cluster_kwargs={"n_workers": 4, "threads_per_worker": 2}
    ),
    log_prints=True,
    retries=1,
    on_failure=[send_failure_notification],
    on_completion=[send_success_notification]
)
async def production_pipeline(
    config: PipelineConfig,
    date_range: tuple,
    transformation_rules: Dict,
    enrichment_api: str
) -> Dict:
    """Production data pipeline with complete feature set."""
    logger = get_run_logger()
    logger.info("Starting production pipeline")

    # Extract data with caching
    raw_data = await extract_data(config, date_range)

    # Validate and transform
    transformed_data = validate_and_transform(raw_data, transformation_rules)

    # Parallel enrichment using batch processing
    batch_size = config.batch_size
    batches = [
        transformed_data[i:i+batch_size]
        for i in range(0, len(transformed_data), batch_size)
    ]

    # Process batches in parallel
    enrichment_futures = [
        enrich_data_batch.submit(batch, enrichment_api)
        for batch in batches
    ]

    # Gather results
    enriched_batches = [f.result() for f in enrichment_futures]
    final_data = pd.concat(enriched_batches, ignore_index=True)

    # Load to destination
    load_result = load_data(final_data, config.destination_bucket)

    # Create final report
    create_markdown_artifact(
        key="pipeline-summary",
        markdown=f"""
        # Pipeline Execution Summary
        - Start Time: {datetime.now() - timedelta(minutes=5)}
        - End Time: {datetime.now()}
        - Records Processed: {load_result['records']}
        - Output Location: {load_result['path']}
        - Status: ✅ Success
        """,
        description="Pipeline execution summary"
    )

    logger.info(f"Pipeline completed: {load_result}")
    return load_result

# Deployment configuration
if __name__ == "__main__":
    # Deploy to production
    production_pipeline.deploy(
        name="daily-data-pipeline",
        work_pool_name="kubernetes-pool",

        # Docker configuration
        image="my-registry/data-pipeline:v1.0.0",
        build=True,
        push=True,

        # Schedule
        cron="0 2 * * *",  # Daily at 2 AM
        timezone="UTC",

        # Parameters
        parameters={
            "config": {
                "source_database": "production_db",
                "destination_bucket": "s3://data-lake/processed",
                "batch_size": 10000,
                "parallel_workers": 8
            },
            "date_range": ["{{ yesterday }}", "{{ today }}"],
            "transformation_rules": {"rule1": "config1"},
            "enrichment_api": "https://api.enrichment.com/v1"
        },

        # Infrastructure
        job_variables={
            "namespace": "data-pipelines",
            "service_account_name": "prefect-runner",
            "resources": {
                "requests": {"memory": "4Gi", "cpu": "2"},
                "limits": {"memory": "8Gi", "cpu": "4"}
            },
            "env": {
                "PREFECT_LOGGING_LEVEL": "INFO",
                "PYTHONUNBUFFERED": "1"
            }
        },

        tags=["production", "daily", "etl", "critical"]
    )
```

## Conclusion

This comprehensive tutorial covers all essential aspects of Prefect v3.4.13 for building production-grade workflow orchestration. The framework's **90% performance improvement**, combined with features like transactional semantics, advanced retry mechanisms, and flexible deployment options, makes it ideal for modern data engineering needs. Key takeaways include:

1. **Rich decorator system** with comprehensive parameters for flows and tasks
2. **Flexible execution patterns** supporting synchronous, asynchronous, and parallel processing
3. **Robust state management** with automatic retries and transaction support
4. **Production-ready deployment** options from local development to Kubernetes
5. **Comprehensive observability** through structured logging and artifacts
6. **Seamless integration** with cloud services and external systems

By following these patterns and best practices, AI agents can implement sophisticated logging and workflow orchestration that scales from simple automation to complex enterprise data platforms.
