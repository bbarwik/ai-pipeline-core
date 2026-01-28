"""Local trace writer for filesystem-based debugging."""

import atexit
import hashlib
import json
import os
import re
import shutil
import socket
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from queue import Empty, Queue
from threading import Lock, Thread
from typing import Any

import yaml

from ai_pipeline_core.logging import get_pipeline_logger

from .config import TraceDebugConfig
from .content import ArtifactStore, ContentWriter

logger = get_pipeline_logger(__name__)


@dataclass
class WriteJob:
    """Job for background writer thread."""

    trace_id: str
    span_id: str
    name: str
    parent_id: str | None
    attributes: dict[str, Any]
    events: list[Any]
    status_code: str  # "OK" | "ERROR" | "UNSET"
    status_description: str | None
    start_time_ns: int
    end_time_ns: int


@dataclass
class SpanInfo:
    """Information about a span for index building."""

    span_id: str
    parent_id: str | None
    name: str
    span_type: str
    status: str
    start_time: datetime
    path: Path  # Actual directory path for this span
    depth: int = 0  # Nesting depth (0 for root)
    order: int = 0  # Global execution order within trace
    end_time: datetime | None = None
    duration_ms: int = 0
    children: list[str] = field(default_factory=list)
    llm_info: dict[str, Any] | None = None
    prefect_info: dict[str, Any] | None = None


@dataclass
class TraceState:
    """State for an active trace."""

    trace_id: str
    name: str
    path: Path
    start_time: datetime
    spans: dict[str, SpanInfo] = field(default_factory=dict)
    root_span_id: str | None = None
    total_tokens: int = 0
    total_cost: float = 0.0
    llm_call_count: int = 0
    span_counter: int = 0  # Global counter for ordering span directories
    merged_wrapper_ids: set[str] = field(default_factory=set)  # IDs of merged wrappers


class LocalTraceWriter:
    """Writes trace spans to local filesystem via background thread.

    Uses a hierarchical directory structure where child spans are nested
    inside parent span directories. Directory names use numeric prefixes
    (01_, 02_, etc.) to preserve execution order when viewed with `tree`.
    """

    def __init__(self, config: TraceDebugConfig):
        """Initialize trace writer with config."""
        self._config = config
        self._queue: Queue[WriteJob | None] = Queue()
        self._traces: dict[str, TraceState] = {}
        self._artifact_stores: dict[str, ArtifactStore] = {}  # One per trace for deduplication
        self._lock = Lock()
        self._shutdown = False

        # Ensure base path exists
        config.path.mkdir(parents=True, exist_ok=True)

        # Clean up old traces if needed
        self._cleanup_old_traces()

        # Start background writer thread
        self._writer_thread = Thread(
            target=self._writer_loop,
            name="trace-debug-writer",
            daemon=True,
        )
        self._writer_thread.start()

        # Register shutdown handler
        atexit.register(self.shutdown)

    def on_span_start(
        self,
        trace_id: str,
        span_id: str,
        parent_id: str | None,
        name: str,
    ) -> None:
        """Handle span start - create directories and record metadata.

        Called from SpanProcessor.on_start() in the main thread.
        Creates hierarchical directories nested under parent spans.
        """
        with self._lock:
            trace = self._get_or_create_trace(trace_id, name)

            # Determine parent path and depth
            if parent_id and parent_id in trace.spans:
                parent_info = trace.spans[parent_id]
                parent_path = parent_info.path
                depth = parent_info.depth + 1
            elif parent_id:
                # Parent ID provided but not found - orphan span, place at root
                logger.warning(
                    f"Span {span_id} has unknown parent {parent_id}, placing at trace root"
                )
                parent_path = trace.path
                depth = 0
            else:
                parent_path = trace.path
                depth = 0

            # Generate ordered directory name (4 digits supports up to 9999 spans)
            trace.span_counter += 1
            safe_name = self._sanitize_name(name)
            dir_name = f"{trace.span_counter:04d}_{safe_name}"

            # Create nested directory
            span_dir = parent_path / dir_name
            span_dir.mkdir(parents=True, exist_ok=True)

            # Record span info
            now = datetime.now(timezone.utc)
            span_info = SpanInfo(
                span_id=span_id,
                parent_id=parent_id,
                name=name,
                span_type="default",
                status="running",
                start_time=now,
                path=span_dir,
                depth=depth,
                order=trace.span_counter,
            )
            trace.spans[span_id] = span_info

            # Track root span
            if parent_id is None:
                trace.root_span_id = span_id

            # Update parent's children list
            if parent_id and parent_id in trace.spans:
                trace.spans[parent_id].children.append(span_id)

            # Append to event log (lightweight - just appends a line)
            self._append_event(
                trace,
                {
                    "type": "span_start",
                    "span_id": span_id,
                    "parent_id": parent_id,
                    "name": name,
                    "path": str(span_dir.relative_to(trace.path)),
                },
            )
            # Note: _write_status() moved to on_span_end for performance
            # (avoids blocking I/O in main thread on every span start)

    def on_span_end(self, job: WriteJob) -> None:
        """Queue span end job for background processing.

        Called from SpanProcessor.on_end() in the main thread.
        """
        if not self._shutdown:
            self._queue.put(job)

    def shutdown(self, timeout: float = 30.0) -> None:
        """Flush queue and stop writer thread."""
        if self._shutdown:
            return
        self._shutdown = True

        # Signal shutdown
        self._queue.put(None)

        # Wait for thread to finish
        self._writer_thread.join(timeout=timeout)

        # Finalize any remaining traces (ones that didn't have root span end yet)
        with self._lock:
            for trace in list(self._traces.values()):
                try:
                    self._finalize_trace(trace)
                except Exception as e:
                    logger.warning(f"Failed to finalize trace {trace.trace_id}: {e}")
            self._traces.clear()

    def _get_or_create_trace(self, trace_id: str, name: str) -> TraceState:
        """Get existing trace or create new one."""
        if trace_id in self._traces:
            return self._traces[trace_id]

        # Create new trace
        timestamp = datetime.now(timezone.utc)
        safe_name = self._sanitize_name(name)
        dir_name = f"{timestamp.strftime('%Y%m%d_%H%M%S')}_{trace_id[:8]}_{safe_name}"
        trace_path = self._config.path / dir_name

        trace_path.mkdir(parents=True, exist_ok=True)
        # Note: No 'spans/' subdirectory - spans are nested hierarchically

        trace = TraceState(
            trace_id=trace_id,
            name=name,
            path=trace_path,
            start_time=timestamp,
        )
        self._traces[trace_id] = trace

        # Create artifact store for this trace
        self._artifact_stores[trace_id] = ArtifactStore(trace_path)

        # Write initial trace metadata
        self._write_trace_yaml(trace)

        # Append trace start event
        self._append_event(
            trace,
            {
                "type": "trace_start",
                "trace_id": trace_id,
                "name": name,
            },
        )

        return trace

    def _writer_loop(self) -> None:
        """Background thread loop for processing write jobs."""
        while True:
            try:
                job = self._queue.get(timeout=1.0)
            except Empty:
                continue

            if job is None:
                # Shutdown signal
                break

            try:
                self._process_job(job)
            except Exception as e:
                logger.warning(f"Trace debug write failed for span {job.span_id}: {e}")

    def _process_job(self, job: WriteJob) -> None:
        """Process a span end job - write all span data."""
        with self._lock:
            trace = self._traces.get(job.trace_id)
            if not trace:
                logger.warning(f"Trace {job.trace_id} not found for span {job.span_id}")
                return

            span_info = trace.spans.get(job.span_id)
            if not span_info:
                logger.warning(f"Span {job.span_id} not found in trace {job.trace_id}")
                return

            span_dir = span_info.path

            # Extract input/output from attributes
            input_content = self._extract_input(job.attributes)
            output_content = self._extract_output(job.attributes)

            # Get artifact store for this trace
            artifact_store = self._artifact_stores.get(job.trace_id)

            # Create content writer with artifact store
            content_writer = ContentWriter(self._config, artifact_store)

            # Write input/output
            input_ref = content_writer.write(input_content, span_dir, "input")
            output_ref = content_writer.write(output_content, span_dir, "output")

            # Extract span type and metadata
            span_type = self._extract_span_type(job.attributes)
            llm_info = self._extract_llm_info(job.attributes)
            prefect_info = self._extract_prefect_info(job.attributes)

            # Update span info (span_info already validated above)
            end_time = datetime.fromtimestamp(job.end_time_ns / 1e9, tz=timezone.utc)
            span_info.end_time = end_time
            span_info.duration_ms = int((job.end_time_ns - job.start_time_ns) / 1e6)
            span_info.status = "failed" if job.status_code == "ERROR" else "completed"
            span_info.span_type = span_type
            span_info.llm_info = llm_info
            span_info.prefect_info = prefect_info

            # Update trace stats
            if llm_info:
                trace.llm_call_count += 1
                trace.total_tokens += llm_info.get("total_tokens", 0)
                trace.total_cost += llm_info.get("cost", 0.0)

            # Build span metadata (input_ref and output_ref are now dicts)
            span_meta = self._build_span_metadata_v3(
                job, input_ref, output_ref, span_type, llm_info, prefect_info
            )

            # Write _span.yaml
            span_yaml_path = span_dir / "_span.yaml"
            span_yaml_path.write_text(
                yaml.dump(span_meta, default_flow_style=False, allow_unicode=True, sort_keys=False),
                encoding="utf-8",
            )

            # Write events.yaml based on config
            if job.events and self._should_write_events(job.status_code):
                events_data = self._format_span_events(job.events)
                events_path = span_dir / "events.yaml"
                events_path.write_text(
                    yaml.dump(events_data, default_flow_style=False, allow_unicode=True),
                    encoding="utf-8",
                )

            # Append to trace event log
            self._append_event(
                trace,
                {
                    "type": "span_end",
                    "span_id": job.span_id,
                    "status": span_info.status if span_info else "unknown",
                    "duration_ms": span_info.duration_ms if span_info else 0,
                },
            )

            # Update index
            self._write_index(trace)

            # Finalize trace when ALL spans are completed (not just root)
            # This handles the case where child span end jobs arrive after root
            running_spans = [s for s in trace.spans.values() if s.status == "running"]
            if not running_spans:
                self._finalize_trace(trace)
                # Remove from memory to prevent memory leak
                del self._traces[job.trace_id]
                if job.trace_id in self._artifact_stores:
                    del self._artifact_stores[job.trace_id]

    def _extract_input(self, attributes: dict[str, Any]) -> Any:
        """Extract input from span attributes."""
        input_str = attributes.get("lmnr.span.input")
        if input_str:
            try:
                return json.loads(input_str)
            except (json.JSONDecodeError, TypeError):
                return input_str
        return None

    def _extract_output(self, attributes: dict[str, Any]) -> Any:
        """Extract output from span attributes."""
        output_str = attributes.get("lmnr.span.output")
        if output_str:
            try:
                return json.loads(output_str)
            except (json.JSONDecodeError, TypeError):
                return output_str
        return None

    def _extract_span_type(self, attributes: dict[str, Any]) -> str:
        """Extract span type from attributes."""
        span_type = attributes.get("lmnr.span.type", "DEFAULT")
        # Map to our types
        type_map = {
            "LLM": "llm",
            "TOOL": "tool",
            "DEFAULT": "default",
        }
        return type_map.get(span_type, "default")

    def _extract_llm_info(self, attributes: dict[str, Any]) -> dict[str, Any] | None:
        """Extract LLM-specific info from attributes."""
        # Check for LLM attributes
        input_tokens = attributes.get("gen_ai.usage.input_tokens") or attributes.get(
            "gen_ai.usage.prompt_tokens"
        )
        output_tokens = attributes.get("gen_ai.usage.output_tokens") or attributes.get(
            "gen_ai.usage.completion_tokens"
        )

        if input_tokens is None and output_tokens is None:
            return None

        return {
            "model": attributes.get("gen_ai.response.model")
            or attributes.get("gen_ai.request.model"),
            "provider": attributes.get("gen_ai.system"),
            "input_tokens": input_tokens or 0,
            "output_tokens": output_tokens or 0,
            "total_tokens": (input_tokens or 0) + (output_tokens or 0),
            "cost": attributes.get("gen_ai.usage.cost", 0.0),
        }

    def _extract_prefect_info(self, attributes: dict[str, Any]) -> dict[str, Any] | None:
        """Extract Prefect-specific info from attributes."""
        run_id = attributes.get("prefect.run.id")
        if not run_id:
            return None

        return {
            "run_id": run_id,
            "run_name": attributes.get("prefect.run.name"),
            "run_type": attributes.get("prefect.run.type"),
            "tags": attributes.get("prefect.tags", []),
        }

    def _build_span_metadata_v3(
        self,
        job: WriteJob,
        input_ref: dict[str, Any],
        output_ref: dict[str, Any],
        span_type: str,
        llm_info: dict[str, Any] | None,
        prefect_info: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Build span metadata dictionary (V3 format with dict refs)."""
        start_time = datetime.fromtimestamp(job.start_time_ns / 1e9, tz=timezone.utc)
        end_time = datetime.fromtimestamp(job.end_time_ns / 1e9, tz=timezone.utc)
        duration_ms = int((job.end_time_ns - job.start_time_ns) / 1e6)

        meta: dict[str, Any] = {
            "span_id": job.span_id,
            "trace_id": job.trace_id,
            "parent_id": job.parent_id,
            "name": job.name,
            "type": span_type,
            "timing": {
                "start": start_time.isoformat(),
                "end": end_time.isoformat(),
                "duration_ms": duration_ms,
            },
            "status": "failed" if job.status_code == "ERROR" else "completed",
        }

        # Add type-specific metadata
        if prefect_info:
            meta["prefect"] = prefect_info

        if llm_info:
            meta["llm"] = llm_info

        # Add content references (input_ref and output_ref are dicts from ContentWriter.write())
        meta["input"] = input_ref
        meta["output"] = output_ref

        # Add error info if failed
        if job.status_code != "OK" and job.status_description:
            meta["error"] = {
                "message": job.status_description,
            }

        return meta

    def _format_span_events(self, events: list[Any]) -> list[dict[str, Any]]:
        """Format span events for YAML output."""
        result = []
        for event in events:
            try:
                event_dict = {
                    "name": event.name,
                    "timestamp": datetime.fromtimestamp(
                        event.timestamp / 1e9, tz=timezone.utc
                    ).isoformat(),
                }
                if event.attributes:
                    event_dict["attributes"] = dict(event.attributes)
                result.append(event_dict)
            except Exception:
                continue
        return result

    def _should_write_events(self, status_code: str) -> bool:
        """Check if events.yaml should be written based on config."""
        mode = self._config.events_file_mode

        if mode == "none":
            return False
        elif mode == "errors_only":
            return status_code == "ERROR"
        elif mode == "all":
            return True
        else:
            # Default to errors_only if unknown mode
            return status_code == "ERROR"

    def _append_event(self, trace: TraceState, event: dict[str, Any]) -> None:
        """Append event to trace event log (JSONL format)."""
        event["ts"] = datetime.now(timezone.utc).isoformat()
        events_path = trace.path / "_events.jsonl"
        with events_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event) + "\n")

    def _write_trace_yaml(self, trace: TraceState) -> None:
        """Write _trace.yaml file."""
        trace_meta = {
            "trace_id": trace.trace_id,
            "name": trace.name,
            "start_time": trace.start_time.isoformat(),
            "end_time": None,
            "duration_seconds": None,
            "status": "running",
            "correlation": {
                "hostname": socket.gethostname(),
                "pid": os.getpid(),
            },
            "stats": {
                "total_spans": len(trace.spans),
                "llm_calls": trace.llm_call_count,
                "total_tokens": trace.total_tokens,
                "total_cost": round(trace.total_cost, 6),
            },
        }

        trace_yaml_path = trace.path / "_trace.yaml"
        trace_yaml_path.write_text(
            yaml.dump(trace_meta, default_flow_style=False, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )

    def _write_index(self, trace: TraceState) -> None:
        """Write split index files: _tree.yaml, _llm_calls.yaml, _errors.yaml."""
        # Sort spans by execution order
        sorted_spans = sorted(trace.spans.values(), key=lambda s: s.order)

        # Write lightweight tree index (always)
        self._write_tree_index(trace, sorted_spans)

        # Write LLM calls index (if enabled)
        if self._config.include_llm_index:
            self._write_llm_index(trace, sorted_spans)

        # Write errors index (if enabled)
        if self._config.include_error_index:
            self._write_errors_index(trace, sorted_spans)

    def _write_tree_index(self, trace: TraceState, sorted_spans: list) -> None:
        """Write _tree.yaml - lightweight tree structure (~5KB)."""
        span_paths: dict[str, str] = {}
        tree_entries = []

        for span in sorted_spans:
            # Skip spans that were identified as wrappers during merge
            if span.span_id in trace.merged_wrapper_ids:
                continue

            relative_path = span.path.relative_to(trace.path).as_posix() + "/"
            span_paths[span.span_id] = relative_path

            # Minimal entry - just hierarchy and navigation
            entry = {
                "span_id": span.span_id,
                "name": span.name,
                "type": span.span_type,
                "status": span.status,
                "path": relative_path,
            }

            # Add parent_id if not root
            if span.parent_id:
                entry["parent_id"] = span.parent_id

            # Add children if any
            if span.children:
                entry["children"] = span.children

            tree_entries.append(entry)

        tree_data = {
            "format_version": 3,
            "trace_id": trace.trace_id,
            "root_span_id": trace.root_span_id,
            "span_count": len(tree_entries),
            "span_paths": span_paths,
            "tree": tree_entries,
        }

        tree_path = trace.path / "_tree.yaml"
        tree_path.write_text(
            yaml.dump(tree_data, default_flow_style=False, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )

    def _write_llm_index(self, trace: TraceState, sorted_spans: list) -> None:
        """Write _llm_calls.yaml - LLM-specific details."""
        llm_calls = []

        for span in sorted_spans:
            if span.llm_info:
                relative_path = span.path.relative_to(trace.path).as_posix() + "/"

                # Get parent context for better identification
                parent_context = ""
                if span.parent_id and span.parent_id in trace.spans:
                    parent_span = trace.spans[span.parent_id]
                    parent_context = f" (in {parent_span.name})"

                llm_entry = {
                    "span_id": span.span_id,
                    "name": span.name + parent_context,  # Add context to distinguish
                    "model": span.llm_info.get("model"),
                    "provider": span.llm_info.get("provider"),
                    "input_tokens": span.llm_info.get("input_tokens", 0),
                    "output_tokens": span.llm_info.get("output_tokens", 0),
                    "total_tokens": span.llm_info.get("total_tokens", 0),
                    "cost": span.llm_info.get("cost", 0.0),
                    "duration_ms": span.duration_ms,
                    "status": span.status,
                    "path": relative_path,
                }

                if span.start_time:
                    llm_entry["start_time"] = span.start_time.isoformat()

                llm_calls.append(llm_entry)

        llm_data = {
            "format_version": 3,
            "trace_id": trace.trace_id,
            "llm_call_count": len(llm_calls),
            "total_tokens": trace.total_tokens,
            "total_cost": round(trace.total_cost, 6),
            "calls": llm_calls,
        }

        llm_path = trace.path / "_llm_calls.yaml"
        llm_path.write_text(
            yaml.dump(llm_data, default_flow_style=False, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )

    def _write_errors_index(self, trace: TraceState, sorted_spans: list) -> None:
        """Write _errors.yaml - failed spans only."""
        error_spans = []

        for span in sorted_spans:
            if span.status == "failed":
                relative_path = span.path.relative_to(trace.path).as_posix() + "/"

                error_entry = {
                    "span_id": span.span_id,
                    "name": span.name,
                    "type": span.span_type,
                    "depth": span.depth,
                    "duration_ms": span.duration_ms,
                    "path": relative_path,
                }

                if span.start_time:
                    error_entry["start_time"] = span.start_time.isoformat()
                if span.end_time:
                    error_entry["end_time"] = span.end_time.isoformat()

                # Get parent chain for context
                parent_chain = []
                current_id = span.parent_id
                while current_id and current_id in trace.spans:
                    parent = trace.spans[current_id]
                    parent_chain.append(parent.name)
                    current_id = parent.parent_id
                if parent_chain:
                    error_entry["parent_chain"] = list(reversed(parent_chain))

                error_spans.append(error_entry)

        if error_spans:  # Only write if there are errors
            errors_data = {
                "format_version": 3,
                "trace_id": trace.trace_id,
                "error_count": len(error_spans),
                "errors": error_spans,
            }

            errors_path = trace.path / "_errors.yaml"
            errors_path.write_text(
                yaml.dump(
                    errors_data, default_flow_style=False, allow_unicode=True, sort_keys=False
                ),
                encoding="utf-8",
            )

    def _detect_wrapper_spans(self, trace: TraceState) -> set[str]:
        """Detect Prefect wrapper spans that should be merged with their inner spans.

        Detection criteria:
        1. Parent has exactly one child
        2. Names match after stripping hash suffix (e.g., "task-abc123" matches "task")
        3. Parent has no I/O (input type is "none")
        4. Parent has prefect.run.id, child does not
        """
        wrappers = set()

        for span_id, span in trace.spans.items():
            # Must have exactly one child
            if len(span.children) != 1:
                continue

            child_id = span.children[0]
            child = trace.spans.get(child_id)
            if not child:
                continue

            # Names must match after stripping hash suffix
            parent_base = re.sub(r"-[a-f0-9]{3,}$", "", span.name)
            child_base = re.sub(r"-[a-f0-9]{3,}$", "", child.name)
            if parent_base != child_base:
                continue

            # Parent must have no I/O (check _span.yaml)
            span_yaml = span.path / "_span.yaml"
            if span_yaml.exists():
                try:
                    span_meta = yaml.safe_load(span_yaml.read_text())
                    if span_meta.get("input", {}).get("type") != "none":
                        continue
                except Exception:
                    continue

            # Parent must have prefect info, child must not
            if not span.prefect_info:
                continue
            if child.prefect_info:
                continue

            wrappers.add(span_id)

        return wrappers

    def _merge_wrapper_spans(self, trace: TraceState) -> None:
        """Merge wrapper spans with their inner spans (virtual merge).

        This modifies the span hierarchy so wrappers are skipped in index output.
        Physical directories remain unchanged - only the logical view changes.
        """
        if not self._config.merge_wrapper_spans:
            return

        wrappers = self._detect_wrapper_spans(trace)
        if not wrappers:
            return

        logger.debug(f"Merging {len(wrappers)} wrapper spans in trace {trace.trace_id}")

        # Cache wrapper IDs for use in tree index writing
        trace.merged_wrapper_ids = wrappers

        # For each wrapper, reparent its child to the wrapper's parent
        for wrapper_id in wrappers:
            wrapper = trace.spans[wrapper_id]
            child_id = wrapper.children[0]
            child = trace.spans[child_id]
            grandparent_id = wrapper.parent_id

            # Update child's parent
            child.parent_id = grandparent_id

            # Update grandparent's children (if grandparent exists)
            if grandparent_id and grandparent_id in trace.spans:
                grandparent = trace.spans[grandparent_id]
                # Remove wrapper, add child
                if wrapper_id in grandparent.children:
                    idx = grandparent.children.index(wrapper_id)
                    grandparent.children[idx] = child_id
            else:
                # Wrapper was root - child becomes new root
                if trace.root_span_id == wrapper_id:
                    trace.root_span_id = child_id

            # Mark wrapper as merged (used in index generation)
            wrapper.children = []  # Clear to indicate it's merged

    def _finalize_trace(self, trace: TraceState) -> None:
        """Finalize a trace - update metadata and generate summary."""
        end_time = datetime.now(timezone.utc)
        duration = (end_time - trace.start_time).total_seconds()

        # Determine final status
        failed_spans = [s for s in trace.spans.values() if s.status == "failed"]
        status = "failed" if failed_spans else "completed"

        # Merge wrapper spans before generating indexes
        self._merge_wrapper_spans(trace)

        # Update _trace.yaml
        trace_meta = {
            "trace_id": trace.trace_id,
            "name": trace.name,
            "start_time": trace.start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration_seconds": round(duration, 2),
            "status": status,
            "correlation": {
                "hostname": socket.gethostname(),
                "pid": os.getpid(),
            },
            "stats": {
                "total_spans": len(trace.spans),
                "llm_calls": trace.llm_call_count,
                "total_tokens": trace.total_tokens,
                "total_cost": round(trace.total_cost, 6),
            },
        }

        trace_yaml_path = trace.path / "_trace.yaml"
        trace_yaml_path.write_text(
            yaml.dump(trace_meta, default_flow_style=False, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )

        # Final index update
        self._write_index(trace)

        # Generate summary if enabled
        if self._config.generate_summary:
            # Lazy import to avoid circular dependency
            from .summary import generate_summary  # noqa: PLC0415

            summary = generate_summary(trace)
            summary_path = trace.path / "_summary.md"
            summary_path.write_text(summary, encoding="utf-8")

        # Append trace end event
        self._append_event(
            trace,
            {
                "type": "trace_end",
                "trace_id": trace.trace_id,
                "status": status,
                "duration_seconds": round(duration, 2),
            },
        )

    def _cleanup_old_traces(self) -> None:
        """Delete old traces beyond max_traces limit."""
        if self._config.max_traces is None:
            return

        # Get all trace directories sorted by modification time
        trace_dirs = []
        for path in self._config.path.iterdir():
            if path.is_dir() and (path / "_trace.yaml").exists():
                trace_dirs.append((path.stat().st_mtime, path))

        trace_dirs.sort(reverse=True)  # Newest first

        # Delete excess traces
        for _, path in trace_dirs[self._config.max_traces :]:
            try:
                shutil.rmtree(path)
            except Exception as e:
                logger.warning(f"Failed to delete old trace {path}: {e}")

    def _sanitize_name(self, name: str) -> str:
        """Sanitize name for safe filesystem use.

        Truncates to 24 chars + 4-char hash to avoid collisions and keep
        paths manageable with deep nesting.
        """
        safe = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name)
        safe = safe.strip(". ")

        # Handle Windows reserved names (CON, PRN, AUX, NUL, COM1-9, LPT1-9)
        reserved = (
            {"CON", "PRN", "AUX", "NUL"}
            | {f"COM{i}" for i in range(1, 10)}
            | {f"LPT{i}" for i in range(1, 10)}
        )
        if safe.upper() in reserved:
            safe = f"_{safe}"

        # Truncate with hash suffix to avoid collisions
        if len(safe) > 28:
            name_hash = hashlib.md5(name.encode()).hexdigest()[:4]
            safe = f"{safe[:24]}_{name_hash}"

        return safe or "span"
