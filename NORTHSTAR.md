# Northstar

## Purpose

This document is the **source of truth** for the architectural decisions behind **ai-pipeline-core**. It explains, in detail, **what we’ve standardized**, **why we chose it**, and **how engineers must work** within these constraints to build reliable, scalable AI processing pipelines across many projects without re-deciding fundamentals.

The goals:

* **One stable platform** you can use in 10+ projects with minimal per-project glue code.
* **High signal, low configuration.** Every major subsystem has a *single* default choice.
* **Predictable operations** from laptop to server: same docker-compose stack locally, same primitives in production.
* **Observability-first AI workflows**: tracing, artifacts, failures, and costs are visible and attributable.
* **Async, typed, minimal** code, in line with the project’s strict standards (see `CLAUDE.md`).

This is not a menu. It’s a **contract**. If you follow it, your pipelines will be easier to write, debug, deploy, and operate. If you don’t, your pipeline may work temporarily—but you’ll be swimming upstream against the platform.

---

## Philosophy

1. **One choice per concern.** Indecision is the enemy of velocity. We pick one and optimize it.
2. **Fast local parity.** The same docker-compose stack runs on a laptop and a small server.
3. **Async by default.** Every I/O path—DB, cache, object store, LLM—must be non-blocking.
4. **Typed, declarative boundaries.** Pydantic models and SQLAlchemy 2.0 keep interfaces explicit.
5. **Observability as a feature.** LMNR spans, Prefect events, and structured logs are mandatory.
6. **Minimal surface area.** We expose a compact set of functions (≤50) that do the right thing.
7. **Artifacts are first-class.** We always know what was produced, where it lives, and how to reproduce it.
8. **Stateless workers.** All state is in Postgres, Redis, S3 (MinIO locally), or Prefect.
9. **Infrastructure symmetry.** Local: docker-compose. Prod: the same containers behind Prefect workers.

---

## Orchestration

### Choice: **Prefect v3** (flows, tasks, deployments)

**Why**

* Clean Python-first ergonomics; first-class async support.
* Robust state machine, artifacts, events, and result storage.
* Rich deployment story (`prefect.yaml`), blocks/variables, and multi-environment profiles.

**How**

* We ship thin wrappers: `@pipeline_task` and `@pipeline_flow`.
* Wrappers **require async** (aligns with our core tenet) and attach LMNR tracing.
* Default `TaskRunner`: **ThreadPoolTaskRunner** (excellent for I/O-heavy AI workloads).
* **No automatic retries** by default (explicit failure is often safer than silent retry storms).
* Use built-in Prefect concurrency tags when needed (`--tag database`) but prefer Redis locks for fine control.

**Rules**

* All long-running/network operations live inside `@pipeline_task` or `@pipeline_flow` and are **async**.
* Use Prefect parameters only for *small* configuration. Large payloads go through S3/MinIO and are referenced by key.
* Flows **return `DocumentList`**; inputs and outputs are always document types, not raw bytes/strings.

**Advantages**

* Less cognitive overhead; consistent runtime model.
* Easy to observe, rerun, or resume at any step.
* Clear separation between control plane (Prefect) and data plane (our storage/DB/cache).

---

## Containers & Deployment

### Choice: **Docker** everywhere + **GHCR** registry

**Why**

* Immutable, reproducible builds.
* Seamless CI with GitHub Actions; built-in GHCR authentication.

**How**

* A single `Dockerfile` in the app builds images for Prefect deployments.
* `prefect.yaml` uses `prefect-docker` steps to build/push to `ghcr.io/<org>/<image>`.
* Workers: Prefect **Docker work pool** (local docker-compose spins up the worker).
* Same image runs locally and in production.

**Rules**

* Images **must be hermetic**: no runtime `pip install`.
* Sensitive values are **not baked into images**—they come from Prefect variables/blocks or `.env` locally.

**Advantages**

* Zero drift from local to production.
* Cached builds; fast rollbacks and reproducible deployments.

---

## Local/Server Parity

### Choice: **docker-compose** stack with Prefect Server, Docker worker, Postgres (pgvector), Redis, MinIO, LiteLLM

**Why**

* A single file runs the entire platform locally or on a small server.
* Provides all infrastructure services with healthchecks and seeded buckets/databases.

**How**

* `docker-compose.yml` includes:

  * **prefect-server** + **prefect-docker-worker**
  * **Postgres** with **pgvector** extension
  * **Redis**
  * **MinIO** + `mc` sidecar to create the default bucket
  * **LiteLLM** proxy
* `.env` file feeds credentials/endpoints; compose interpolates them.

**Rules**

* Don’t point your laptop directly at production resources.
* Treat local MinIO and Postgres as disposable (but persistent across restarts via Docker volumes).

**Advantages**

* Minutes to first run.
* High confidence that deployments will behave the same way in production.

---

## Database

### Choice: **PostgreSQL** (async via SQLAlchemy 2.0 + `asyncpg`) with **Alembic** migrations

**Why**

* The most reliable, widely-supported OSS relational database.
* **SQLAlchemy 2.0** async engine is mature and fast.
* Declarative metadata, type safety, and battle-tested migrations via Alembic.

**How**

* The framework provides `ai_pipeline_core.db`:

  * `session.py`: engine creation, async session factory, connection pooling tuned for worker workloads.
  * `models.py`: baseline tables (pipeline runs, document metadata, artifact registry, idempotency keys).
  * `alembic/versions`: owned migrations packaged with the library (semver-aligned).
* `aipctl db upgrade` runs the included migrations against the configured database.
* **pgvector** lives in the same Postgres instance (see “Vector Search”).

**Rules**

* Never emit raw SQL in application code; always use SQLAlchemy Core/ORM.
* Migrations are run **before** any flow is deployed (CI step or `aipctl up` sequence).
* For large JSON blobs, prefer storing in S3 and a small metadata row in Postgres, not giant JSON columns.

**Advantages**

* Safe schema evolution across 10+ projects.
* One pooling strategy to tune and one failure mode to learn.

---

## Vector Search

### Choice: **pgvector** (in-database embeddings)

**Why**

* Simplifies deployment: one DB for transactional + vector workloads.
* Strong locality for metadata joins (no cross-system consistency issues).
* Good performance for millions of vectors with HNSW/IVFFlat indexes.

**How**

* Alembic migration creates `vector` extension, embeddings table, and ANN index.
* Framework exposes:

  * `embed(texts: list[str]) -> list[vector]` (via LiteLLM),
  * `upsert_embeddings(items)`,
  * `similarity_search(query_vector, filter_fields, k)`.
* Batching + backoff; optional Redis rate limiting per provider key.

**Rules**

* Use **one** embedding model globally (`text-embedding-3-large` via LiteLLM).
* Keep embeddings **idempotent**: stable primary key per document chunk.
* Store only the vector and minimal metadata (doc\_id, chunk\_id, tags); keep the heavy content in S3.

**Advantages**

* Single operational surface.
* Fewer moving parts than a separate vector DB, easier backups and access control.

---

## Object Storage

### Choice: **S3 API** with **MinIO locally**, **AWS S3 in prod**

**Why**

* S3 is ubiquitous for large binary and tabular artifacts.
* MinIO provides a faithful S3 API locally.

**How**

* A single **Prefect `S3Bucket` block** is used in code, configured with endpoint + credentials:

  * Local: `http://minio:9000`, path-style addressing forced, `minioadmin` creds.
  * Prod: AWS S3 with IAM.
* A helper standardizes **key layout**:

  ```
  {project}/{flow}/{run_ts_utc_iso}/{step}/{artifact_name}
  ```
* Artifact helpers:

  * `write_parquet(df, key)`, `read_parquet(key)` (pyarrow),
  * `write_json(obj, key)`, `read_json(key)`,
  * `write_bytes(bytes, key)`, `read_bytes(key)`.

**Rules**

* No ad-hoc paths—always use the key layout helper.
* Parquet is the **default** for tabular artifacts; JSON for light structured metadata; raw bytes for binary.
* Keep flow results and artifacts in S3; store references in Postgres if needed for cross-run linking.

**Advantages**

* Clear, human-navigable artifact trees.
* Zero surprises when debugging locally vs. prod.

---

## Cache, Coordination & Rate Limiting

### Choice: **Redis**

**Why**

* Simple, fast, and reliable for ephemeral control: locks, idempotency, rate limiting, and small queues.

**How**

* Framework provides `ai_pipeline_core.cache.redis`:

  * `get_async_client()` connection factory,
  * `async_lock(key, ttl)`,
  * `idempotency_key(key, ttl)` (set-if-not-exists),
  * `rate_limit(bucket, limit, window_seconds)` (token bucket or sliding window).
* Used to:

  * prevent duplicate work when a flow is retried manually,
  * guard external APIs (LiteLLM, webhooks),
  * debounce notifications.

**Rules**

* Prefer Redis locks for *cross-task mutual exclusion* over database row locks.
* Keep TTLs tight; Redis is not a database.

**Advantages**

* Avoids thundering herds and API bans.
* Easy, explicit concurrency control shared across the whole worker fleet.

---

## DataFrames & Serialization

### Choice: **pandas** + **pyarrow** with **Parquet** as the default format

**Why**

* De facto standard; easiest developer ergonomics.
* Parquet + pyarrow gives interop with Spark, DuckDB, and big-data tools.

**How**

* Artifact helpers read/write Parquet from S3 transparently (streaming when sensible).
* For small structured configs, we use JSON or YAML via documents.

**Rules**

* Don’t pass giant DataFrames through Prefect parameters; write them to S3 and pass keys.
* Normalize columns/typing before persistence (no mixed types).

**Advantages**

* Performance, compatibility, and predictable costs.

---

## ORM & Migrations

### Choice: **SQLAlchemy 2.0 (async)** + **Alembic**

**Why**

* Modern async API; typed, explicit `select()`/`insert()` semantics.
* Alembic has the right blend of automation and control.

**How**

* Session management via `async_sessionmaker`, `async with session.begin()`.
* Alembic revision per semver bump if DB schema changes.

**Rules**

* One engine, one session factory, many sessions.
* Use Core/ORM constructs; no string SQL in application code.

**Advantages**

* Safe, readable queries.
* Upgrades you can trust.

---

## LLM Access

### Choice: **LiteLLM proxy** for all model calls

**Why**

* Single interface to many providers; consistent headers, cost metadata, and retries.
* Local `litellm` container mirrors prod proxy config; no API surface differences.

**How**

* `OPENAI_BASE_URL` points to LiteLLM; we use OpenAI-compatible Chat API in code.
* `generate()` and `generate_structured()` already in the framework (with LMNR spans).
* We add `embed()` to call `text-embedding-3-large` via LiteLLM.

**Rules**

* Never call providers directly from application code.
* Set capacity/rate limits in Redis per provider key when needed.

**Advantages**

* Switch models/providers without changing application code.
* Uniform observability and error handling.

---

## Embeddings

### Choice: **`text-embedding-3-large`** (via LiteLLM)

**Why**

* High-quality embeddings with broad ecosystem support.
* Fixing one model preserves consistency across pipelines and projects.

**How**

* `embed(texts: list[str])` batches requests (size tuned by token/latency measurements), with backoff and jitter.
* Returns vectors ready for pgvector insert/upsert.
* Embedding results are cached with a **content hash** in Redis to avoid recomputation.

**Rules**

* A change in embedding model is a breaking change for vector search; if you change it, migrate or rebuild vectors.
* Keep chunking stable across runs (document/chunk IDs must be deterministic).

**Advantages**

* Repeatable similarity across environments and time.
* Lower infra and cognitive load.

---

## Observability

### Choice: **LMNR** for tracing + **Prefect logs/events**; zero “print” logging.

**Why**

* Structured spans tell you what happened, where, and at what cost.
* Prefect logs tie spans to flow runs; artifacts and events provide forensic breadcrumbs.

**How**

* `@trace` decorator wraps all tasks/flows.
* We enrich spans with:

  * model, tokens, cached tokens, cost (from LiteLLM headers),
  * artifact keys, doc IDs, chunk counts,
  * DB timings, row counts, retry attempts.
* Prefect `log_prints=True` captures output, but **we don’t print**; we use structured logger helpers.

**Rules**

* No direct `logging`; use `get_pipeline_logger`.
* Attach context: `document_id`, `artifact_key`, `flow_name`, `project`, etc.
* Report failures with enough context to reproduce.

**Advantages**

* Fewer mystery outages.
* Cheaper debug cycles—time goes into fixing, not guessing.

---

## Configuration & Secrets

### Choice: **Pydantic Settings + Prefect Variables/Blocks**, `.env` only for local

**Why**

* Strongly-typed config with defaulting and validation.
* Central, auditable configuration in Prefect for non-local environments.

**How**

* `Settings` reads:

  1. Prefect Variables (when available),
  2. `.env` (local),
  3. Environment variables.
* Blocks for:

  * S3 bucket (MinIO/AWS),
  * Docker registry credentials,
* Variables for:

  * `DATABASE_URL`, `REDIS_URL`, `PUSHOVER_USER/TOKEN`,
  * `OPENAI_BASE_URL` (LiteLLM), `OPENAI_API_KEY` (LiteLLM key),
  * Project defaults.

**Rules**

* No secrets in code, images, or repo.
* For local development, `.env` is acceptable but never committed.

**Advantages**

* Secure by default; same code runs everywhere.

---

## Notifications

### Choice: **Pushover** (with a pluggable notifier interface)

**Why**

* Simple HTTP API, reliable mobile notifications.
* Easy to add Slack/Email later without changing call sites.

**How**

* `Notifier` protocol (`send(title, message, **kwargs)`).
* `PushoverNotifier` implementation using async `httpx`.
* Centralized **on\_failure** hooks in Prefect call the notifier with flow/task metadata.
* Redis dedupe (e.g., suppress duplicate alerts for the same failure within 5 minutes).

**Rules**

* Use failure hooks; don’t sprinkle ad-hoc notifiers inside tasks.
* Keep alert messages concise but actionable: flow, run ID, link to Prefect UI, artifact keys.

**Advantages**

* Minimal noise; fast signal.
* Extensible channel strategy without touching business logic.

---

## Artifacts & Results

### Choice: **S3/MinIO for results and artifacts**, **Postgres for metadata**, **Redis for ephemeral**

**Why**

* Binary data and DataFrames belong in object storage, not DBs.
* DB holds the graph of what exists and where to find it.

**How**

* Key layout `(project/flow/run_ts/step/name)` is enforced by helpers.
* Prefect result storage uses the **same** S3 block.
* Postgres tables map artifacts to flows and runs for quick lookup/search.

**Rules**

* All persisted data must be retrievable with a stable key or index.
* Never store large blobs in Postgres.

**Advantages**

* Cheap storage, crisp lookups, easy clean-up policies.

---

## Error Policy

### Choice: **No automatic retries by default**

**Why**

* Blind retries can amplify problems (rate limits, quotas, bad inputs).
* We prefer **explicit** retries where they make sense.

**How**

* `@pipeline_task` defaults to zero retries.
* Expose helper(s) for common retry policies (exponential backoff with jitter) to opt in task-by-task.
* For external APIs:

  * network/transient errors → opt-in backoff,
  * 4xx errors → **do not retry** (unless specifically allowed).

**Rules**

* Decide at task definition time whether retries are appropriate.
* Check idempotency before enabling retries (e.g., writes must be safe to re-run).

**Advantages**

* Failures are noticed early; less hidden flakiness.
* Operational behavior is explicit and reviewable.

---

## Idempotency

### Choice: **Redis keys for idempotency**, **document content hashes**

**Why**

* With manual retries or repeated triggers, duplicate work must be avoided cheaply.

**How**

* `idempotency_key("task:<hash>")` backing a short TTL prevents duplicates.
* Document IDs are **first 6 chars of content SHA-256 in base32**, used in artifact keys and spans.

**Rules**

* Any task doing non-reversible external effects must be idempotent or protected by locks and keys.
* Use deterministic chunk IDs for vector upsert.

**Advantages**

* Safer reruns; fewer accidental double-charges on third-party APIs.

---

## CLI

### Choice: **`aipctl`**—one command with focused subcommands

**Why**

* Repeatable, discoverable workflows for all projects.
* Keeps the function surface small (well under the 30–50 cap).

**How** *(representative subset)*

* `aipctl init` – create `.env`, Prefect profile, bootstrap blocks, show URLs.
* `aipctl up` – bring up docker-compose & wait for health.
* `aipctl check` – connectivity to DB, Redis, MinIO, LiteLLM, Prefect.
* `aipctl db upgrade` – run migrations.
* `aipctl blocks init` – create/update S3/Docker blocks, set variables.
* `aipctl deploy` – `prefect deploy` using the repository’s `prefect.yaml`.
* `aipctl run <flow>` – run a flow locally or via Prefect (flagged).

**Rules**

* Don’t write ad-hoc scripts for common tasks; add a subcommand instead.
* Keep CLI UX stable; breaking changes must bump major version.

**Advantages**

* One muscle memory across all repos.
* Lower onboarding time for new engineers.

---

## Security

**Principles**

* No secrets in code/images.
* Minimum necessary IAM (S3), network segmentation for DB/Redis.
* Service tokens for CI/CD and workers, rotated periodically.

**Practices**

* `.env` only for local; production via Prefect variables/blocks or secret managers.
* S3 bucket policies scoped per environment.
* Postgres users per environment; no superusers for app.

**Advantages**

* Reasonable hardening with minimal ceremony.
* Clear responsibilities between app code and platform.

---

## Testing & Quality

**Testing**

* All tests async; no network calls in unit tests.
* Integration tests run against the compose stack (labels: `integration`).
* Migrations are tested (create → upgrade → seed → queries).

**Quality Gates**

* Ruff + basedpyright enforced via pre-commit.
* Coverage ≥80% (core modules).
* CI: build, lint, typecheck, test, package.

**Advantages**

* Scale across projects without drift.
* Fail fast on correctness and style regressions.

---

## Versioning & Releases

**Choice:** **SemVer** with automated GH Actions

* Tag `vX.Y.Z` triggers:

  * Build and push GHCR image.
  * Build wheels/sdist; publish to PyPI.
  * (Optional) `prefect deploy` step.

**Rules**

* API changes to the public surface bump **minor** (additive) or **major** (breaking).
* Schema changes: ship Alembic migration in the same release.

**Advantages**

* Consumers can pin or float safely.
* Repeatable deployment loops across all projects.

---

## Extensibility

**Principle:** pluggable adapters, single implementations shipped

* Today we ship Pushover; tomorrow we can add Slack without changing call sites.
* Today we ship pgvector; tomorrow we *could* add a feature-flagged alternative—but we won’t until a compelling reason emerges.

**Rules**

* Add new backends via explicit interfaces and factory functions.
* Never add a second default. Defaults must remain singular.

**Advantages**

* Pragmatism with room to grow.
* Protects the ecosystem from choice paralysis.

---

## Anti-Patterns (Don’t Do This)

* **Multiple DBs or vector stores** in a single project (“one more for embeddings”). Use pgvector.
* **Raw provider API calls** that bypass LiteLLM—loses retries, metrics, headers, and consistency.
* **Giant Prefect parameters**—pass S3 keys, not 200MB blobs.
* **Ad-hoc notification calls** sprinkled in tasks—use central failure hooks.
* **Printing**—use the structured logger, not `print`.
* **Implicit retries everywhere**—they hide systemic issues; opt in intentionally.
* **Large JSON storage in Postgres**—store binary/tabular in S3; keep Postgres for metadata.
* **Dynamic imports, global mutable state, or sync I/O**—violates core rules.

---

## Operational Playbook

**Bring up the stack**

1. `aipctl init` → creates `.env`, Prefect profile, blocks, and bucket.
2. `aipctl up` → starts compose; waits for DB/Redis/MinIO/LiteLLM/Prefect health.
3. `aipctl db upgrade` → applies migrations.
4. `aipctl check` → smoke test.

**Build & deploy**

1. `git tag vX.Y.Z` → CI builds GHCR image and PyPI package.
2. `aipctl deploy` (or CI) → updates Prefect deployment.

**Debug a failure**

1. Pushover notification arrives with flow/run link.
2. Open Prefect UI: inspect logs, artifacts (S3 keys), and LMNR spans.
3. If data issue, fetch artifact from S3 via the key helper.
4. Fix; re-run from failed step or entire flow.
5. If repeated failures—evaluate whether retries (opt-in) make sense **and** add idempotency locks.

**Scale**

* Increase Docker worker concurrency; set Prefect deployment concurrency tags.
* Rate limit LLM calls via Redis if needed.
* Introduce batching at the task level (embedding/chunking).

---

## Rationale Recap (Quick Justifications)

* **Prefect v3**: Pythonic control plane, async-friendly, deployments + artifacts.
* **Docker + GHCR**: reproducibility + CI best-fit.
* **Postgres + SQLAlchemy + Alembic**: safe transactional base; migrations at scale.
* **pgvector**: one DB for vectors and metadata; fewer moving parts.
* **MinIO/S3**: artifact growth without headaches; same code paths locally and in prod.
* **Redis**: the right tool for locks, rate limits, and idempotency.
* **pandas/pyarrow/Parquet**: efficient, compatible, and ergonomic for tabular artifacts.
* **LiteLLM + `text-embedding-3-large`**: provider abstraction with a single, high-quality embedding standard.
* **LMNR + structured logs**: traceable pipelines with costs and inputs captured.
* **No default retries**: explicit operational behavior; fewer silent problems.
* **Strict async + typing**: performance and correctness as a habit, not an afterthought.

---

## How to Work Within These Choices (Rules of Engagement)

* **Define documents, not blobs.** All flow inputs/outputs are `Document` subclasses; no loose strings/bytes.
* **Persist artifacts deliberately.** Use the storage helper; never invent your own key scheme.
* **Write to one of three places only:** S3 for big data, Postgres for relations/metadata, Redis for ephemeral control.
* **Use our LLM helpers exclusively.** They add tracing and uniform metadata; do not call SDKs directly.
* **Annotate types and keep everything async.** If it blocks, it’s a bug.
* **Prefer composition.** If you need a new capability (e.g., notify Slack), add a small adapter that implements the existing interface.
* **Keep flows small.** A flow coordinates tasks and I/O; business logic hides in tasks and pure functions.
* **Fail loudly, then decide if retries belong.** Don’t paper over bad inputs or upstream failures.
* **Document migrations and bump versions** when schemas or function signatures change.

---

## Closing

The value of **ai-pipeline-core** isn’t just the code—it’s the **discipline** captured in these choices. By committing to a sharp, opinionated stack and the rules above, we gain:

* Faster delivery across many projects with a shared mental model.
* Lower operational toil and simpler on-call—same logs, same artifacts, same dashboards.
* Better reliability: fewer hidden states, fewer dead ends, faster post-mortems.

This **Northstar** doc is the contract we keep with ourselves and our future collaborators. If a change is truly necessary, update the doc first—explain **why** the old constraint no longer serves us—then change the code. Until then, lean into the defaults. They’re designed to keep you **shipping**, not re-architecting.
