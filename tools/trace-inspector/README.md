# trace-inspector

Reads from `ai-pipeline-core`'s database layer (FilesystemDatabase or ClickHouse) and generates a self-contained markdown bundle for debugging pipeline runs.

## Installation

Part of the workspace. Installed with the dev dependencies:

```bash
uv pip install --system -e tools/trace-inspector/
```

Provides the `ai-trace-inspect` CLI command.

## Usage

### Inspect a full run

```bash
# From a downloaded FilesystemDatabase snapshot
ai-trace-inspect run --db-path ./downloaded_bundle/ --out ./inspect/

# From ClickHouse by run ID
ai-trace-inspect run --run-id my-project --out ./inspect/

# Only failed tasks
ai-trace-inspect run --db-path ./downloaded_bundle/ --out ./inspect/ --failed

# Restrict to one flow
ai-trace-inspect run --db-path ./downloaded_bundle/ --out ./inspect/ --flow-name AnalysisFlow
```

### Inspect one flow

```bash
ai-trace-inspect flow --db-path ./downloaded_bundle/ --out ./inspect/ --flow-name AnalysisFlow
ai-trace-inspect flow --db-path ./downloaded_bundle/ --out ./inspect/ --flow-span-id 550e8400-...
```

### Inspect one task with provenance neighbors

```bash
ai-trace-inspect task --db-path ./downloaded_bundle/ --out ./inspect/ --task-span-id 550e8400-...
```

### Compare a task across two traces (original vs replay)

```bash
ai-trace-inspect compare \
  --left-db-path ./original/ \
  --right-db-path ./replay/ \
  --out ./inspect/ \
  --left-task-span-id 550e8400-...
```

### Download from ClickHouse first, then inspect locally

```bash
ai-trace-inspect run --run-id my-project --download-db-to ./snapshot/ --out ./inspect/
```

## Output structure

```
inspect/
  index.md              # Run overview, flow table, failure summary
  docs/
    ABCD1234.md         # One file per document (content + provenance)
  flows/
    01_analysis_flow_F001/
      flow.md           # Flow summary with task table
      tasks/
        01_extract_task_T001/
          task.md        # Task detail: conversations, documents, provenance
        02_summarize_batch_T002-T010/
          batch.md       # Collapsed batch for repeated leaf tasks
```

Each task page includes: information flow (input/output documents with provenance), LLM conversation transcripts, sub-task links, and diagnosis for failures.

## Programmatic API

```python
from trace_inspector import inspect_run, compare_runs, Selection, SourceSpec, RenderConfig

selection = Selection(
    mode="run",
    source=SourceSpec(db_path=Path("./downloaded_bundle/")),
    output_dir=Path("./inspect/"),
)
await inspect_run(selection)
```
