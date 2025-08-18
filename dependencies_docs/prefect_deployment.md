# Prefect v3.4.13 comprehensive technical tutorial

This comprehensive tutorial covers Prefect version 3.4.13, focusing on environment variables, settings, API keys, secrets, deployment, configuration, and logging implementation. The guide assumes you have experience with Python programming and are implementing logging functionality.

## Installation and initial setup

Prefect v3.4.13 requires Python 3.9 or newer. The recommended installation approach uses a virtual environment to isolate dependencies.

### Primary installation methods

```bash
# Create virtual environment
python -m venv prefect-env
source prefect-env/bin/activate  # On Windows: prefect-env\Scripts\activate

# Install Prefect via pip
pip install prefect

# Verify installation
prefect version
```

For modern package management with uv:

```bash
# Install uv package manager
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create environment and install Prefect
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -U prefect
```

### Docker deployment setup

```yaml
# docker-compose.yml
version: '3.8'
services:
  prefect-server:
    image: prefecthq/prefect:3-latest
    command: prefect server start --host 0.0.0.0
    environment:
      - PREFECT_API_URL=http://prefect-server:4200/api
    ports:
      - "4200:4200"
    networks:
      - prefect-network

networks:
  prefect-network:
    driver: bridge
```

### Connecting to Prefect Cloud

```bash
# Interactive cloud login
prefect cloud login

# Or set API credentials directly
export PREFECT_API_KEY="pnu_your-api-key"
export PREFECT_API_URL="https://api.prefect.cloud/api/accounts/[account-id]/workspaces/[workspace-id]"
```

## Environment variables and configuration hierarchy

Prefect v3.4.13 uses a hierarchical configuration system where settings cascade from multiple sources. Understanding this hierarchy is crucial for proper configuration management.

### Configuration precedence order

Settings are applied in the following order (highest to lowest precedence):
1. Environment variables (prefixed with `PREFECT_`)
2. `.env` file in current directory
3. `prefect.toml` or `pyproject.toml` files
4. Profile settings
5. Default values

### Core environment variables

```bash
# API Configuration
export PREFECT_API_URL="http://127.0.0.1:4200/api"
export PREFECT_API_KEY="pnu_your-api-key"
export PREFECT_API_AUTH_STRING="admin:password"  # For self-hosted servers
export PREFECT_API_REQUEST_TIMEOUT=60
export PREFECT_API_TLS_INSECURE_SKIP_VERIFY=false

# Logging Configuration
export PREFECT_LOGGING_LEVEL="INFO"
export PREFECT_LOGGING_LOG_PRINTS=true
export PREFECT_LOGGING_TO_API_ENABLED=true
export PREFECT_LOGGING_COLORS=true
export PREFECT_LOGGING_MARKUP=false

# Flow and Task Configuration
export PREFECT_FLOWS_DEFAULT_RETRIES=0
export PREFECT_FLOWS_DEFAULT_RETRY_DELAY_SECONDS=0
export PREFECT_TASKS_DEFAULT_RETRIES=0
export PREFECT_TASKS_REFRESH_CACHE=false

# Results and Storage
export PREFECT_RESULTS_DEFAULT_SERIALIZER="pickle"
export PREFECT_RESULTS_PERSIST_BY_DEFAULT=false
export PREFECT_LOCAL_STORAGE_PATH="~/.prefect/storage"

# Worker Configuration
export PREFECT_WORKER_HEARTBEAT_SECONDS=30
export PREFECT_WORKER_QUERY_SECONDS=10
export PREFECT_WORKER_PREFETCH_SECONDS=10
```

### Profile management

Profiles enable switching between different environments easily:

```bash
# Create environment-specific profiles
prefect profile create development
prefect profile create staging
prefect profile create production

# Configure production profile
prefect profile use production
prefect config set PREFECT_API_URL="https://api.prefect.cloud/api"
prefect config set PREFECT_API_KEY="pnu_production_key"
prefect config set PREFECT_LOGGING_LEVEL="WARNING"

# Switch between profiles
prefect profile use development

# View current configuration
prefect config view --show-defaults
```

Profile configurations are stored in `~/.prefect/profiles.toml`:

```toml
[profiles.development]
PREFECT_API_URL = "http://localhost:4200/api"
PREFECT_LOGGING_LEVEL = "DEBUG"

[profiles.production]
PREFECT_API_URL = "https://api.prefect.cloud/api/accounts/{account_id}/workspaces/{workspace_id}"
PREFECT_API_KEY = "pnu_production_key"
PREFECT_LOGGING_LEVEL = "INFO"
```

## API keys and authentication

Prefect supports multiple authentication methods depending on your deployment model.

### API key types

- **User API Keys** (prefix: `pnu_`): Associated with individual user accounts
- **Service Account API Keys** (prefix: `pnb_`): For automated systems (Pro/Enterprise)

### Authentication configuration

```python
from prefect.client.orchestration import PrefectClient

# Using environment variables (recommended)
client = PrefectClient()

# Explicit configuration
client = PrefectClient(
    api_url="https://api.prefect.cloud/api",
    api_key="pnu_your-api-key"
)

# Test connection
async def test_connection():
    async with client as c:
        account_info = await c.read_account_info()
        print(f"Connected to account: {account_info.name}")
```

### Custom headers for proxy authentication

```bash
export PREFECT_CLIENT_CUSTOM_HEADERS='{"Proxy-Authorization": "Bearer proxy-token", "X-Corporate-ID": "corp-id"}'
```

## Secrets management

Prefect v3.4.13 provides comprehensive secrets management through its block system with encrypted storage and external integration capabilities.

### Creating and using secret blocks

```python
from prefect.blocks.system import Secret
from pydantic import SecretStr
from prefect.blocks.core import Block

# Simple secret creation
secret = Secret(value="my-secret-password")
secret.save("database-password")

# Custom block with secure fields
class DatabaseCredentials(Block):
    host: str
    port: int = 5432
    username: str
    password: SecretStr  # Automatically obfuscated

# Usage in flows
from prefect import flow, task

@task
def connect_to_database():
    secret = Secret.load("database-password")
    password = secret.get()
    # Use password for connection
    return "Connected"

@flow
def data_pipeline():
    connect_to_database()
```

### Integration with external secret managers

#### AWS Secrets Manager

```python
from prefect_aws import AwsSecret, AwsCredentials

@flow
def aws_secret_flow():
    aws_secret_block = AwsSecret.load("my-aws-secret")
    secret_value = aws_secret_block.read_secret()

    # Use secret in your workflow
    return process_with_secret(secret_value)
```

#### HashiCorp Vault

```python
from prefect_vault import VaultSecret

@flow
async def vault_integration_flow():
    vault_secret = await VaultSecret.load('my-vault-secret')
    secret_value = vault_secret.get_secret('secret/data/myapp')['api_key']
    return f"Using API key: {secret_value[:10]}..."
```

### Environment-specific secret management

```python
import os
from prefect import flow

@flow
def environment_aware_flow():
    env = os.environ.get("ENVIRONMENT", "development")

    # Load environment-specific secrets
    secret_name = f"{env}-db-credentials"
    db_creds = DatabaseCredentials.load(secret_name)

    # Use credentials
    return connect_with_credentials(db_creds)
```

## Deployment configuration with prefect.yaml

The `prefect.yaml` file defines deployment specifications and build steps.

### Complete prefect.yaml structure

```yaml
# Generic metadata
prefect-version: null
name: my-project

# Build steps for Docker
build:
  - prefect_docker.deployments.steps.build_docker_image:
      id: build-image
      requires: prefect-docker>=0.6.0
      image_name: "{{ prefect.variables.registry }}/{{ prefect.variables.image_name }}"
      tag: "{{ prefect.variables.image_tag }}"
      dockerfile: auto
      platform: "linux/amd64"

# Push to registry
push:
  - prefect_docker.deployments.steps.push_docker_image:
      image_name: "{{ build-image.image_name }}"
      tag: "{{ build-image.tag }}"
      credentials: "{{ prefect.blocks.docker-registry-credentials.production }}"

# Runtime steps
pull:
  - prefect.deployments.steps.git_clone:
      repository: "{{ prefect.variables.repository_url }}"
      branch: "{{ prefect.variables.branch }}"
      credentials: "{{ prefect.blocks.github_credentials.my-github-token }}"

# Deployment configurations
deployments:
  - name: production-deployment
    version: null
    tags: ["production", "data-pipeline"]
    description: "Production data pipeline"
    schedule:
      cron: "0 2 * * *"
      timezone: "UTC"
    entrypoint: src/main.py:main_flow
    parameters:
      environment: production
      batch_size: 1000
    work_pool:
      name: kubernetes-production
      work_queue_name: default
      job_variables:
        image: "{{ build-image.image }}"
        namespace: production
        memory: 4096
        cpu: "2"
        env:
          DATABASE_URL: "{{ prefect.blocks.secret.db-url }}"
          LOG_LEVEL: "INFO"
```

### Initialization and deployment

```bash
# Initialize deployment configuration
prefect init --recipe docker

# Deploy specific deployment
prefect deploy --name production-deployment

# Deploy all deployments
prefect deploy --all

# Non-interactive deployment for CI/CD
prefect deploy --all --no-prompt
```

## Work pools and workers

Work pools manage the execution infrastructure for deployments.

### Creating and configuring work pools

```bash
# Create Kubernetes work pool
prefect work-pool create k8s-production --type kubernetes

# Create Docker work pool
prefect work-pool create docker-production --type docker

# Set concurrency limits
prefect work-pool set-concurrency-limit k8s-production 10
```

### Kubernetes work pool configuration

```yaml
# k8s-work-pool-template.yaml
apiVersion: batch/v1
kind: Job
metadata:
  labels: "{{ labels }}"
  namespace: "{{ namespace }}"
  generateName: "{{ name }}-"
spec:
  ttlSecondsAfterFinished: "{{ finished_job_ttl }}"
  template:
    spec:
      parallelism: 1
      completions: 1
      restartPolicy: Never
      serviceAccountName: "{{ service_account_name }}"
      containers:
      - name: prefect-job
        image: "{{ image }}"
        command: "{{ command }}"
        env: "{{ env }}"
        resources:
          requests:
            memory: "{{ memory }}Mi"
            cpu: "{{ cpu }}"
          limits:
            memory: "{{ memory_limit }}Mi"
            cpu: "{{ cpu_limit }}"
```

### Starting workers

```bash
# Start Docker worker
prefect worker start --pool docker-production --type docker

# Start Kubernetes worker with environment variables
PREFECT_API_KEY=your_key \
PREFECT_API_URL=https://api.prefect.cloud/api/accounts/your-account/workspaces/your-workspace \
prefect worker start --pool k8s-production --type kubernetes
```

## Logging implementation

Prefect's logging system is built on Python's standard logging framework with extensive customization capabilities.

### Basic logging configuration

```python
from prefect import flow, task
from prefect.logging import get_run_logger

@task
def logging_task():
    logger = get_run_logger()

    logger.info("Starting task execution")
    logger.debug("Debug information")
    logger.warning("Warning message")
    logger.error("Error occurred")

    # Structured logging with extra fields
    logger.info(
        "Processing batch",
        extra={
            "batch_id": "batch_001",
            "record_count": 1000,
            "processing_time": 45.2
        }
    )

@flow(log_prints=True)  # Capture print statements
def logging_flow():
    logger = get_run_logger()
    logger.info("Flow started")
    print("This print will be captured as a log")
    logging_task()
```

### Custom logging configuration

Create `~/.prefect/logging.yml`:

```yaml
version: 1
disable_existing_loggers: False

formatters:
  standard:
    format: "%(asctime)s.%(msecs)03d | %(levelname)-7s | %(name)s | %(message)s"
    datefmt: "%Y-%m-%d %H:%M:%S"

  json:
    class: pythonjsonlogger.jsonlogger.JsonFormatter
    format: "%(asctime)s %(levelname)s %(name)s %(message)s"

handlers:
  console:
    class: prefect.logging.handlers.PrefectConsoleHandler
    level: INFO
    formatter: standard

  api:
    class: prefect.logging.handlers.APILogHandler
    level: INFO

  file:
    class: logging.handlers.RotatingFileHandler
    level: DEBUG
    formatter: standard
    filename: prefect.log
    maxBytes: 10485760  # 10MB
    backupCount: 5

loggers:
  prefect:
    level: INFO
    handlers: [console, file]

  prefect.flow_runs:
    level: INFO
    handlers: [console, api, file]

  prefect.task_runs:
    level: INFO
    handlers: [console, api, file]
```

### External logging integration

```python
import logging
from prefect.logging.handlers import PrefectLogHandler

class DataDogHandler(logging.Handler):
    def __init__(self, api_key, app_key):
        super().__init__()
        self.api_key = api_key
        self.app_key = app_key

    def emit(self, record):
        log_entry = {
            'timestamp': record.created * 1000,
            'level': record.levelname.lower(),
            'message': record.getMessage(),
            'service': 'prefect',
            'source': 'python'
        }

        # Add Prefect metadata
        if hasattr(record, 'flow_run_id'):
            log_entry['tags'] = [f"flow_run_id:{record.flow_run_id}"]

        self._send_to_datadog(log_entry)

class CloudWatchHandler(logging.Handler):
    def __init__(self, log_group, log_stream, aws_region='us-east-1'):
        super().__init__()
        self.log_group = log_group
        self.log_stream = log_stream

    def emit(self, record):
        import boto3

        cloudwatch = boto3.client('logs', region_name=self.aws_region)

        log_event = {
            'timestamp': int(record.created * 1000),
            'message': self.format(record)
        }

        cloudwatch.put_log_events(
            logGroupName=self.log_group,
            logStreamName=self.log_stream,
            logEvents=[log_event]
        )
```

### Performance monitoring with logging

```python
from prefect import flow, task
from prefect.logging import get_run_logger
import time
import functools

def performance_monitor(func):
    """Decorator to monitor task performance"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = get_run_logger()
        start_time = time.time()

        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time

            logger.info(
                f"Task {func.__name__} completed",
                extra={
                    "execution_time": execution_time,
                    "status": "success"
                }
            )
            return result

        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(
                f"Task {func.__name__} failed",
                extra={
                    "execution_time": execution_time,
                    "error": str(e),
                    "status": "failed"
                }
            )
            raise

    return wrapper

@task
@performance_monitor
def monitored_task(data):
    # Task implementation
    return process_data(data)
```

## Production deployment strategies

### Kubernetes deployment

```python
from prefect import flow

@flow
def kubernetes_flow():
    # Flow implementation
    pass

if __name__ == "__main__":
    kubernetes_flow.deploy(
        name="k8s-production-deployment",
        work_pool_name="kubernetes-production",
        job_variables={
            "image": "myregistry/flow:v1.2.3",
            "namespace": "production",
            "memory": 2048,
            "cpu": "1000m",
            "env": {
                "DATABASE_URL": "{{ prefect.blocks.secret.db-url }}",
                "LOG_LEVEL": "INFO"
            }
        },
        schedule={
            "cron": "0 2 * * *",
            "timezone": "UTC"
        }
    )
```

### Docker deployment with CI/CD

```yaml
# .github/workflows/deploy.yml
name: Deploy Prefect Flows

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4

    - name: Setup Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'

    - name: Install Prefect
      run: pip install prefect prefect-docker

    - name: Deploy flows
      env:
        PREFECT_API_KEY: ${{ secrets.PREFECT_API_KEY }}
        PREFECT_API_URL: ${{ secrets.PREFECT_API_URL }}
      run: prefect deploy --all --no-prompt
```

### High availability configuration

```yaml
# Kubernetes HPA for workers
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: prefect-worker-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: prefect-worker
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

## Best practices for production

### Security configuration checklist

- Store all secrets in dedicated secret management systems
- Use service accounts for automated systems
- Enable audit logging for compliance
- Rotate API keys and credentials regularly
- Implement network policies for worker access
- Use TLS encryption for all communications

### Monitoring and observability

```python
from prefect import flow, task
from prefect.blocks.notifications import SlackWebhook

@flow
def monitored_production_flow():
    slack = SlackWebhook.load("production-alerts")

    try:
        result = main_processing()
        return result
    except Exception as e:
        slack.notify(f"Flow failed: {str(e)}")
        raise

# Configure monitoring thresholds
ALERT_THRESHOLDS = {
    "flow_failure_rate": 0.05,  # 5% failure rate
    "avg_duration_increase": 2.0,  # 2x baseline
    "worker_cpu_utilization": 0.90,  # 90% CPU
    "queue_backlog": 20,  # 20 pending runs
}
```

### Performance optimization

```python
from prefect import flow, task
from prefect.task_runners import ConcurrentTaskRunner
from datetime import timedelta

@task(
    cache_key_fn=task_input_hash,
    cache_expiration=timedelta(hours=1),
    retries=3,
    retry_delay_seconds=exponential_backoff(backoff_factor=2)
)
def optimized_task(data):
    # Cached, retryable task with exponential backoff
    return process_data(data)

@flow(
    task_runner=ConcurrentTaskRunner(),
    persist_result=True
)
def optimized_flow():
    # Concurrent execution with result persistence
    results = optimized_task.map(data_batches)
    return results
```

### Project structure recommendation

```
my-prefect-project/
├── prefect.yaml              # Deployment configuration
├── .prefect/                 # Prefect metadata
├── flows/
│   ├── __init__.py
│   ├── etl_flow.py
│   └── ml_pipeline.py
├── tasks/
│   ├── __init__.py
│   ├── data_processing.py
│   └── notifications.py
├── blocks/
│   └── storage_blocks.py
├── deployments/
│   └── deploy_flows.py
├── logging.yml              # Custom logging configuration
├── requirements.txt
├── Dockerfile
├── .env                     # Environment variables
└── README.md
```

## Conclusion

This tutorial provides comprehensive coverage of Prefect v3.4.13's core features for production deployments. The combination of flexible configuration management, robust secrets handling, extensive logging capabilities, and multiple deployment strategies makes Prefect suitable for enterprise-grade workflow orchestration. Focus on implementing proper security practices, comprehensive monitoring, and efficient resource utilization to ensure successful production deployments.
