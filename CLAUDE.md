# AI Pipeline Core Framework Requirements

> Internal requirements for the ai-pipeline-core framework repository. This document defines what the framework must provide and how framework code must be written.

## Design Principles

1. **Framework Absorbs Complexity, Apps Stay Simple** — All heavy/complex logic lives in the framework. Application code built on this framework should be minimal and straightforward. Tracing, retries, deployment, progress tracking, storage, logging, and validation are handled automatically.

2. **Deploy First, Optimize Later** — Get working system fast. Architecture must allow future optimization without major refactoring.

3. **Distributed by Default** — Multiple processes on independent machines with centralized services (LiteLLM, ClickHouse, logging). Design to avoid race conditions and data duplication.

4. **AI-Native Development** — Designed for AI coding agents to understand, modify, and debug. One correct way to do everything. Definition-time validation catches mistakes before runtime.

5. **Single Source of Truth** — No duplicate documentation. Code defines behavior. Auto-generate documentation from code.

6. **Self-Explanatory Code** — Code must be understandable without deep-diving into documentation or framework source code. Naming, structure, and types make intent obvious.

7. **Automate Everything Possible** — If a check, validation, or transformation can be automated, it must be. Manual steps invite errors.

8. **Minimal Code** — Less code is better code. Every line must justify its existence.

9. **No Legacy Code** — No backward compatibility layers, deprecation shims, or references to previous implementations. Unused code must be removed immediately.

10. **No Unvalidatable Derivatives** — When a value is derived from a typed source (field name, class name, enum variant), it must be computed programmatically, not written as a manual string. Dict keys mirroring model fields, string identifiers mirroring class names — if the type checker can't trace it back to the source, derive it from the typed source instead. This prevents silent breakage when renaming.

---

## 1. Core Architecture

### 1.1 Async Execution

All operations must be asynchronous. No blocking I/O calls allowed.

**`async def` must contain async operations** — Functions declared with `async def` must contain at least one `await`, `async for`, or `async with` statement. Functions without async operations must not be marked `async`. Enforced via semgrep rule.

**Exceptions:**
- Protocol stubs (method signature only)
- ABC base class methods meant for override
- In-memory test implementations (e.g., MemoryDocumentStore)

### 1.2 Immutability & Safety

- No mutable global variables (constants or initialized-once immutable globals only)
- Default timeouts on all operations — nothing hangs indefinitely
- Strict type checking throughout
- Pydantic models use `frozen=True` where possible
- Dataclasses use `frozen=True` and `slots=True`

### 1.3 Module System

- Modules/agents/tools form **acyclic dependency graph**
- Clear module boundaries with defined inputs/outputs
- All inputs and outputs must be JSON serializable
- **Context as typed schema** — Context documents have defined types with role, citation policy, and usage description
- Missing required context documents caught at assembly time, not runtime

### 1.4 Configuration

- System-level config via `Settings` base class (Pydantic BaseSettings)
- Module-level overrides when needed
- No config duplication — define once, reuse everywhere
- Model configuration includes model name AND model-specific options (e.g., `reasoning_effort`)

---

## 2. What the Framework Provides

This section documents capabilities the framework must provide to applications.

### 2.1 LLM Interaction (`llm/`, `_llm_core/`)

**Core Functions:**
- `generate()` — Async LLM generation with automatic retries, tracing, cost tracking
- `generate_structured()` — Structured output with Pydantic BaseModel response format

**Conversation Class:**
- Immutable, stateful conversation management
- Separates **context** (cacheable prefix: documents, system instructions) from **messages** (dynamic suffix: conversation history)
- **Immutability**: Every method returns a NEW `Conversation` instance. The original is never modified. This enables safe forking for warmup+fork pattern.
- **Methods**: `send(content)`, `send_structured(content, response_format)`, `with_document(doc)`, `with_context(*docs)`, `restore_content(text)`
- **Properties**: `.content`, `.reasoning_content`, `.usage`, `.cost`, `.parsed`, `.citations`

**Prompt Caching:**
- Automatic prefix-based caching via `cache_ttl` parameter (default: 5 minutes)
- `context_count` parameter — Number of messages from the beginning that form the cacheable prefix
- Warmup + fork pattern supported (see §3.2 for implementation details)
- **WARNING: Gemini requires ≥10k tokens in context for caching to be effective**

**Image Handling:**
- Automatic processing for images exceeding model-specific limits
- `ImagePreset` enum: GEMINI (3000x3000, 9M pixels), CLAUDE (1568x1568, 1.15M pixels), GPT4V (2048x2048, 4M pixels)
- Tall images: split vertically into tiles with 20% overlap
- Wide images: trimmed (left-aligned crop)
- Format conversion (GIF, WEBP, RGBA → JPEG/PNG)
- A single image consumes **1080 tokens** regardless of pixel dimensions

**Model Options:**
- `ModelOptions` with: `reasoning_effort`, `cache_ttl`, `retries`, `retry_delay_seconds`, `timeout`, `service_tier`, `search_context_size`
- `system_prompt` — System-level instructions for the model
- `temperature` — Optional, default to None (provider decides). Only set if you have a specific reason.
- `max_completion_tokens` — Controls output length. Default: provider decides (~30K typical).

**Resilience:**
- Configurable retries with exponential backoff
- Rate limit handling with graceful backoff
- Model fallbacks (primary → secondary → tertiary) — configured via LiteLLM
- Provider fallbacks (OpenAI → Gemini → Grok) — configured via LiteLLM

**Content Protection:**
- `ContentSubstitutor` — Protocol defining substitution interface (`prepare`, `substitute`, `restore`)
- `URLSubstitutor` — Concrete implementation handling URLs, blockchain addresses, and high-entropy strings
- Entropy-based detection identifies strings that are likely tokens/keys
- Enabled by default in `Conversation`; use `restore_content()` to restore originals in outputs

### 2.2 Documents (`documents/`)

**Document Class:**
- Immutable Pydantic model wrapping binary content with metadata
- SHA256 content-addressed storage (deduplication)
- MIME type detection (extension-based + content analysis)
- Automatic content conversion: `str | bytes | dict | list | BaseModel` → bytes

**Provenance Tracking:**
- `sources` — Content provenance (document SHA256 hashes or URLs)
- `origins` — Causal provenance (document SHA256 hashes only)
- Validation: same SHA256 cannot appear in both sources and origins

**Definition-Time Validation (`__init_subclass__`):**
- Rejects class names starting with "Test" (pytest conflict)
- Rejects custom fields beyond allowed set (`name`, `description`, `content`, `sources`, `origins`, `attachments`)
- Detects canonical name collisions between Document subclasses
- Validates `FILES` enum if defined

**Attachments:**
- `Attachment` class for multi-part documents (e.g., markdown + screenshot + PDF)
- Primary content in `Document.content`, secondary in `attachments` tuple

### 2.3 Pipeline Decorators (`pipeline/`)

**@pipeline_task:**
- Wraps async functions with Prefect `@task` + Laminar tracing
- Automatic document persistence to DocumentStore
- Document lifecycle tracking (created, returned, orphaned)
- Configurable retries, timeouts, trace levels

**@pipeline_flow:**
- Wraps async functions with Prefect `@flow` + Laminar tracing
- Extracts input/output document types from annotations at decoration time
- Validates: exactly 3 parameters, correct types, no input/output type overlap
- Sets RunContext for document storage scoping

**Decoration-Time Validation:**
- Return type must be `Document | None | list[Document] | tuple[Document, ...]`
- Rejects mixed types (e.g., `str | Document`)
- Rejects invalid containers (e.g., `dict[str, Document]`)

**Graceful Degradation:**
- Some modules/tasks can be marked as optional (allowed to fail)
- Failed optional modules logged as warning, not error
- Workflow continues with reduced functionality when appropriate

### 2.4 Document Storage (`document_store/`)

**DocumentStore Protocol:**
- `save()`, `save_batch()` — Idempotent by SHA256
- `load()` — Load documents by type and run scope
- `check_existing()` — Check which SHA256s exist
- `has_documents()` — Check if documents of type exist

**Implementations:**
- `ClickHouseDocumentStore` — Production (indexed, circuit breaker)
- `LocalDocumentStore` — CLI/debug (filesystem, browsable)
- `MemoryDocumentStore` — Testing (dict-based)

### 2.5 Deployment (`deployment/`)

**PipelineDeployment Base Class:**
- Generic typing: `PipelineDeployment[TOptions, TResult]`
- Per-flow resume via DocumentStore (skip if outputs exist)
- Per-flow uploads (not just at pipeline end)
- Webhook progress reporting (per-flow start/completion)
- CLI interface with `--start`/`--end` step control
- Prefect deployment generation via `as_prefect_flow()`

**Progress Tracking:**
- `progress_update()` for intra-flow progress
- Webhook push + Prefect labels pull
- Weighted calculation based on `estimated_minutes`

**Hooks System:**
- `DeploymentHook` ABC for external packages
- Artifact injection and environment variable merging

### 2.6 Agents (`agents/`)

**AgentProvider ABC:**
- `run()` — Execute agent with inputs, return `AgentResult`
- `list_agents()` — Optional, returns available agents
- `validate()` — Optional, validates configuration

**AgentResult:**
- Immutable container: `success`, `output`, `artifacts`, `error`, `traceback`
- Artifact access: `get_artifact()`, `get_artifact_bytes()`

**AgentOutputDocument:**
- Wraps `AgentResult` as Document with enforced provenance
- `origins` parameter required (raises if empty)

**Registry:**
- Thread-safe singleton pattern
- `register_agent_provider()`, `get_agent_provider()`, `run_agent()`
- `temporary_provider()` context manager for testing

**Tool Categories:**
- **Authoritative tools** — Return verified data with sources. Can be used as provenance.
- **Advisory tools** — Help but can fail; not usable as sources (e.g., search models without citations)
- Clear distinction affects how tool outputs are tracked for provenance

### 2.7 Observability (`observability/`)

**Tracing:**
- `@trace` decorator with Laminar integration
- Automatic input/output capture
- Document content trimming for traces (text: first/last 100 chars, binary: removed)
- TraceInfo propagation across function calls
- **Prompt traceability** — Every LLM call logs the context used
- Local debug trace system with structured execution summaries (`.trace` directory)

**Document Tracking:**
- Lifecycle events: CREATED, TASK_INPUT, TASK_OUTPUT, LLM_CONTEXT, LLM_MESSAGE
- ClickHouse storage for spans, documents, costs (optional dependency — no-op when not configured)
- Optional LLM-generated summaries for spans and documents (cheap model, background thread)

**Logging:**
- `get_pipeline_logger()` — Configured logger with context
- Logging bridge forwards to OTel span events
- **Correlation IDs** for distributed tracing
- **Automatic context capture** — worker, module, function captured automatically

---

## 3. LLM Implementation Rules

These rules govern how the framework's LLM module must be implemented.

### 3.1 Token Economics — Input Tokens Are Cheap

**Core principle:** Input tokens are cheap; cached input tokens are near-free. Never sacrifice accuracy or context quality to reduce input size. Full context improves accuracy and is cheap.

The framework must:
- Support large contexts (100K-250K tokens)
- **Never trim or summarize inputs** to "save tokens"
- Implement prefix-based caching with configurable TTL
- Prefer sending identical large prefixes across calls over sending tailored smaller prompts per call

### 3.2 Warmup + Fork Pattern

When multiple LLM calls share the same context, the framework must support the warmup + fork pattern:

1. **Warmup call**: Send shared context with a short warmup message. This populates the cache.
2. **Capture warmup response**: The LLM acknowledges. **This response must be kept** — it becomes part of the shared prefix for all forks.
3. **Fork**: Create N parallel calls. Each fork's messages start with the warmup conversation (warmup message + warmup response), then append per-fork content.

**Critical**: The warmup response is essential. Without it, fork messages diverge immediately after the warmup message — the provider only caches `context + warmup message` (a few hundred tokens). With it, the cached prefix includes `context + warmup message + warmup response`. Discarding the warmup response defeats the purpose of the warmup call.

**Timing constraint**: All forked calls must be sent within 5 minutes of the warmup call (cache TTL).

### 3.3 Preparation-First Execution

Because cache lives at most 5 minutes, the framework must support preparation-first execution:

1. **Fetch phase**: Gather all external data (web content, screenshots, API responses)
2. **Warmup phase**: Send shared context to LLM
3. **Execution phase**: Fire all forked LLM calls simultaneously

**Anti-pattern**: Interleaving slow I/O with LLM calls causes cache misses because later calls may arrive after cache TTL expires.

### 3.4 No Batching

Do not batch multiple items into a single LLM call to "save tokens." With caching:
- Separate calls are nearly as cheap as batched calls (shared prefix is cached)
- Separate calls produce higher accuracy (LLM focuses on one task)
- Separate calls are easier to implement, retry, and debug
- Separate calls return structured output per item without complex parsing

### 3.5 Context vs Messages Split

LLM calls have two parts:
- **Context** — Static, cacheable prefix (documents, system instructions)
- **Messages** — Dynamic suffix (conversation history, task instructions)

The framework must:
- Maintain clear separation in `Conversation` class
- Support `context_count` parameter to define cacheable prefix
- Apply cache control directives to context messages

### 3.6 Image Handling

Maximum resolution varies by model preset: Gemini 3000x3000, Claude 1568x1568, GPT-4V 2048x2048.

**Image Processing Pipeline:**
1. **Load and normalize** — EXIF orientation fix (important for mobile photos)
2. **If within model limits** — Encode as JPEG, send as single image
3. **If taller than limit** — Split vertically into tiles with **20% overlap**, each tile within height limit
4. **If wider than limit** — Trim width (left-aligned crop). Web content is left-aligned, so right-side content is typically less important.
5. **Describe the split in text prompt** — "Screenshot was split into N sequential parts with overlap"

The framework must:
- Process images according to model-specific `ImagePreset` limits
- Apply **20% overlap** between vertical tiles to prevent content loss at boundaries
- Convert unsupported formats (GIF, WEBP) to JPEG/PNG
- Prefer larger tiles to minimize token cost

### 3.7 Model Options

The framework exposes `ModelOptions` with:
- `reasoning_effort` — For thinking models ("low", "medium", "high")
- `cache_ttl` — Prompt caching duration (default: "300s")
- `retries`, `retry_delay_seconds`, `timeout`
- `service_tier` — OpenAI tier selection (other providers ignore this)
- `search_context_size` — For search-enabled models
- `system_prompt` — System-level instructions for the model
- `temperature` — Optional, default to None (provider decides)
- `max_completion_tokens` — Output length limit (default: provider decides, ~30K typical)

### 3.8 Model Cost Tiers

The framework should guide apps toward cost-effective model selection:

- **Expensive** (pro/flagship): gemini-3-pro, gpt-5.1. Use for complex reasoning, final synthesis.
- **Cheap** (flash/fast): gemini-3-flash, grok-4.1-fast. Use for high-volume tasks, formatting, conversion, structured output extraction.
- **Too small** (nano/lite): Insufficient for production pipeline tasks. Do not use.

### 3.9 Model Reference Preservation

**Do not remove model references that appear unfamiliar.** Models are released frequently and the codebase may reference models that are newer than the AI coding agent's training data. If a model name exists in code, assume it is valid unless there is concrete evidence otherwise (e.g., provider returns "model not found" error).

### 3.10 Structured Output

The framework's `generate_structured()` must:
- Accept Pydantic BaseModel as `response_format`
- Send schema to LLM automatically (never explain JSON structure in prompts)
- Parse and validate response against model
- Return `ModelResponse[T]` with `.parsed` accessor

**Quality limits**:
- Structured outputs degrade beyond ~2-3K tokens
- Nesting beyond 2 levels causes quality degradation
- `dict` types are not supported in structured output — use lists of typed models
- Complex structures should be split across multiple calls

**Decomposition Fields Before Decision Fields:**
In BaseModel definitions, fields that decompose the problem into concrete dimensions must be defined before fields that represent conclusions. LLMs generate tokens sequentially — if the decision field comes first, the LLM commits to a conclusion and then rationalizes it.

```python
# WRONG — decision before analysis
class VerificationResult(BaseModel):
    is_valid: bool
    summary: str

# WRONG — generic scratchpad
class VerificationResult(BaseModel):
    reasoning: str  # Just "think step by step" in a field
    is_valid: bool

# CORRECT — domain-specific decomposition leads to decision
class VerificationResult(BaseModel):
    source_content_summary: str   # What the source says
    report_claims: str            # What the report claims
    discrepancies: str            # Differences found
    assessment: str               # Reasoned conclusion
    is_valid: bool                # Decision follows from decomposition
```

### 3.11 Document XML Wrapping

When documents are added to LLM context, the framework wraps them:

```xml
<document>
  <id>A7B2C9</id>
  <name>report.md</name>
  <description>Research report</description>
  <content>
  [full document text]
  </content>
</document>
```

This XML boundary separates data from instructions — the **prompt injection defense**. The system prompt instructs the LLM to treat document content as data, not executable instructions.

### 3.12 Thinking Models

All LLMs (2026) perform internal reasoning. The framework must NOT add:
- Chain-of-thought prompting
- "Think step by step" instructions
- Scratchpad patterns

These are redundant and can interfere with the model's native reasoning. Reasoning effort is controlled via `ModelOptions.reasoning_effort` where supported.

### 3.13 Long Response Handling

LLMs produce quality degradation in responses longer than 3-5K tokens. The framework must support:
- Conversational follow-up patterns for building long outputs incrementally
- Multi-turn exchanges where each follow-up receives previous responses as conversation history

Tasks requiring long outputs should not use a single call requesting a large response.

### 3.14 LLM Anti-Patterns

The framework must **not** implement or encourage these patterns:

| Anti-Pattern | Why It's Wrong |
|--------------|----------------|
| Batching multiple items in one call | Caching makes separate calls nearly as cheap; accuracy degrades with batching |
| Generic `reasoning: str` scratchpad fields | Redundant with model's native reasoning; use domain-specific decomposition fields |
| Chain-of-thought prompting | All 2026 models are thinking models; explicit CoT is redundant |
| Numeric confidence scores without criteria | Each call interprets scale differently; hallucinated results |
| Trimming inputs to "save tokens" | Input tokens are cheap; context improves accuracy |
| Explaining JSON structure in prompts | Redundant with schema sent via `response_format`; degrades quality |

---

## 4. Code Quality Standards

### 4.1 Type Safety

- Complete type hints on all functions and return values
- Pydantic models for all data structures
- Use specific types: `UUID` not `str` for identifiers, `Path` not `str` for file paths
- Constrained strings must be custom types (NewType, Annotated, or wrapper class)
- Definition-time validation via `__init_subclass__` where applicable

### 4.2 Testing

- Tests serve as usage examples
- Test mode allows running with cheaper/faster models (simple model swap via config)
- Individual modules must be testable in isolation
- Framework provides test harness utilities requiring zero configuration

**Test-First Bug Fixing:**
When a bug is discovered, write a failing test first. The test must fail, proving the bug exists. Only then implement the fix. After fixing, the test must pass.

### 4.3 AI-Focused Documentation

The framework auto-generates documentation for AI coding agents via `docs_generator/`:
- Public/private determined by `_` prefix convention
- Full source code with comments included
- Examples extracted from test suite
- 50KB warning threshold per guide
- CI-enforced freshness

**Visibility by Naming Convention:**
- No `_` prefix → public (included in docs)
- Single `_` prefix → private (excluded)
- Dunder methods (`__init__`, `__eq__`, etc.) → always public
- Files starting with `_` (e.g., `_helpers.py`) → private modules (excluded entirely)
- Exception: `__init__.py` is always processed

**Docstring Rules:**
- No `Example:` blocks — tests serve as examples
- Constraint keywords (`cannot`, `must`, `never`, `always`, `critical`) at line start are extracted to RULES section
- Inline comments within method bodies are preserved
- Constraint keyword must be the **first word** after optional whitespace

**Test Marking:**
- `@pytest.mark.ai_docs` — Explicitly include a test as an example
- Marked tests get priority — included first regardless of score
- Auto-selected tests are scored by: symbol overlap (high bonus), test length (shorter preferred), mock usage (penalty)
- Error examples using `pytest.raises` included in ERROR EXAMPLES section

**Internal Types:**
- Private classes matching `_CapitalizedName` pattern that appear in public API signatures are automatically included in INTERNAL TYPES section
- Ensures guides are self-contained

### 4.4 Module Cohesion

Each framework module produces one AI-docs guide. That guide must be **self-sufficient for usage**: an AI coding agent must be able to correctly use the module's public API by reading only that guide.

**The acid test**: "Can an AI agent correctly use this module by reading only its guide?" If using module A requires reading module B's guide, the module boundaries must be redrawn.

- **One concern, one module** — Related functionality lives in a single module directory
- **Public API self-documentation** — Parameters triggering behavior in other modules must be documented on the public API
- **Imports allowed, knowledge dependencies forbidden** — Module A may import from B internally, but using A's public API must not require reading B's documentation

---

## 5. Code Hygiene Rules

### 5.1 Protocol and Implementation Separation

Protocol definitions must not be mixed with implementations in the same file. When a Protocol is needed, place it in a separate `_protocols.py` or `_types.py` module.

**Exceptions:** Files explicitly excluded in semgrep config (protocol.py, base.py, _types.py).

### 5.2 No Patch Reference Comments

Comments referencing bug fixes (`# FIX 1:`, `# Fixes #123`, `# Fixes issue`, `# Patch for...`, `# Workaround for...`) are forbidden. Code must be self-explanatory. Bug fixes are documented by regression tests.

### 5.3 Magic Number Constants

Numeric literals used as thresholds, limits, or configuration values must be defined as module-level or class-level constants with descriptive names.

**Exceptions:** `0`, `1`, `-1`, `2`, and standard mathematical constants.

```python
# Wrong
if len(url) < 40:
    return url

# Correct
MIN_URL_LENGTH_FOR_SHORTENING = 40
```

### 5.4 Silent Exception Handling

`except Exception: pass` and `except: pass` are forbidden. Caught exceptions must be:
1. Logged with context, OR
2. Re-raised (possibly wrapped), OR
3. Converted to a specific return value with a comment explaining why swallowing is safe

### 5.5 File Size Limits

- **Warning:** Files exceeding 500 lines (excluding blanks and comments)
- **Error:** Files exceeding 1000 lines

**Suggested splits:**
- Types/protocols → `_types.py` or `_protocols.py`
- Pure functions/utilities → `_utils.py`
- Constants/patterns → `_constants.py`

### 5.6 Export Discipline

Every module with public symbols must define `__all__` listing its public API. Internal modules must be prefixed with `_` (e.g., `_helpers.py`).

### 5.7 Module Naming

Module and directory names must describe the domain problem, not implementation technique.

**Anti-pattern:** `content_protection/` (describes technique)
**Correct:** `token_reduction/` or `url_shortener/` (describes purpose)

### 5.8 Algorithm Complexity

Operations on unbounded input should prefer O(n) or O(n log n). O(n²) is acceptable only when:
- Input size has a known small bound (e.g., n ≤ 100), AND
- The simpler algorithm reduces code complexity

Document size assumptions when using higher-complexity algorithms.

### 5.9 Duplicate Logic

Functions or match/case blocks with >80% structural similarity must be consolidated. Use parameterization, helper functions, or lookup tables.

---

## 6. Deployment & Operations

### 6.1 Deployment Safety

- New deployments must not break running workflows
- Running processes finish on old version
- New requests use new version
- Graceful version transition

### 6.2 Scalability

- Horizontal scaling via additional workers
- Centralized services handle coordination (LiteLLM, ClickHouse)
- No single points of failure (where possible)
- Deployment system should be able to manage resources (API keys, models, scaling)

---

## 7. Decisions Made

| Decision | Choice | Notes |
|----------|--------|-------|
| Orchestrator | Prefect | Flow/task orchestration, state management |
| LLM Proxy | LiteLLM (primary), OpenRouter (compatible) | Unified multi-provider access |
| Document Storage | ClickHouse + local | Content-addressed with SHA256 deduplication |
| Tracking Storage | ClickHouse | Time-series for spans, documents, costs |
| Tracing | Laminar (LMNR) + OpenTelemetry | Span-based with automatic instrumentation |

---

## 8. Not Yet Implemented

These features are planned but not implemented:

| Feature | Description |
|---------|-------------|
| Anomaly Detection | Detect model hangs, response loops, malformed responses, incorrect outputs |
| Prompt Compilation | Static compiler, runtime renderer, dynamic compiler. Three-stage architecture: Static Compiler (dev time, AI-heavy), Runtime Renderer (deterministic), Dynamic Compiler (runtime AI). |
| MCP Server-Side Execution | Server-side multi-turn to avoid prompt re-sends. Highly cost-optimized by avoiding prompt re-sends (even cached tokens cost money). |
| Checkpoint Granularity | Module-level, task-level, or call-level recovery |
| Three-Tier Parameter System | Context Documents (schema at def, content at runtime), Static Parameters (compile time), Dynamic Parameters (runtime) |

---

## 9. Out of Scope

- Compliance/regulatory features (GDPR, SOC2)
- Multi-tenant isolation
- Complex access control/RBAC
- Custom orchestrator implementation

## 10. Bash Guidelines

### IMPORTANT: Avoid commands that cause output buffering issues
- DO NOT pipe output through `head`, `tail`, `less`, or `more` when monitoring or checking command output
- DO NOT use `| head -n X` or `| tail -n X` to truncate output - these cause buffering problems
- Instead, let commands complete fully, or use `--max-lines` flags if the command supports them
- For log monitoring, prefer reading files directly rather than piping through filter

### When checking command output:
- Run commands directly without pipes when possible
- If you need to limit output, use command-specific flags (e.g., `git log -n 10` instead of `git log | head -10`)
- Avoid chained pipes that can cause output to buffer indefinitely
