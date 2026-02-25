"""Local trace writer for filesystem-based debugging."""

import atexit
import json
import re
import shutil
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from queue import Empty, Queue
from threading import Lock, Thread
from typing import Any, ClassVar

import yaml

from ai_pipeline_core.logging import get_pipeline_logger

from ._config import SpanInfo, TraceDebugConfig, TraceState, WriteJob
from ._content import ContentWriter
from ._summary import generate_costs, generate_summary

logger = get_pipeline_logger(__name__)

_MAX_SPAN_NAME_LENGTH = 20


@dataclass(frozen=True, slots=True)
class _FlushRequest:
    """Sentinel queued to force the writer to drain all preceding jobs."""

    done: threading.Event


class LocalTraceWriter:
    """Writes trace spans to local filesystem via background thread.

    Uses a hierarchical directory structure where child spans are nested
    inside parent span directories. Directory names use numeric prefixes
    (01_, 02_, etc.) to preserve execution order when viewed with `tree`.
    Generates index files and optionally produces summary.md for trace analysis.
    """

    def __init__(self, config: TraceDebugConfig):
        """Initialize trace writer with config."""
        self._config = config
        self._queue: Queue[WriteJob | _FlushRequest | None] = Queue()
        self._traces: dict[str, TraceState] = {}
        self._lock = Lock()
        self._shutdown = False

        # Ensure base path exists
        config.path.mkdir(parents=True, exist_ok=True)

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
                logger.warning("Span %s has unknown parent %s, placing at trace root", span_id, parent_id)
                parent_path = trace.path
                depth = 0
            else:
                parent_path = trace.path
                depth = 0

            # Generate ordered directory name: NNN_name matching summary tree format (NNN-name)
            trace.span_counter += 1
            safe_name = self._sanitize_name(name)
            counter = trace.span_counter
            # 3 digits by default, grows to 4+ if needed
            width = 3 if counter < 1000 else len(str(counter))
            dir_name = f"{counter:0{width}d}_{safe_name}"

            # Create nested directory
            span_dir = parent_path / dir_name
            span_dir.mkdir(parents=True, exist_ok=True)

            # Record span info
            now = datetime.now(UTC)
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

    def on_span_end(self, job: WriteJob) -> None:
        """Queue span end job for background processing.

        Called from SpanProcessor.on_end() in the main thread.
        """
        if not self._shutdown:
            self._queue.put(job)

    def record_filtered_llm_metrics(self, trace_id: str, attributes: dict[str, Any]) -> None:
        """Record cost/token metrics from a filtered LLM span for trace totals.

        Only counts document_summary spans — the only non-Conversation caller of _llm_core.
        Conversation-based LLM spans have a parent DEFAULT span that carries the same metrics.
        """
        purpose = attributes.get("purpose", "")
        if not isinstance(purpose, str) or not purpose.startswith("document_summary:"):
            return
        with self._lock:
            trace = self._traces.get(trace_id)
            if not trace:
                return
            llm_info = self._extract_llm_info(attributes)
            if not llm_info:
                return
            trace.llm_call_count += 1
            trace.total_tokens += llm_info.get("total_tokens", 0)
            trace.total_cost += llm_info.get("cost", 0.0)

    def flush(self, timeout: float = 30.0) -> bool:
        """Block until all queued jobs are processed.

        Returns True if the flush completed within the timeout, False otherwise.
        """
        if self._shutdown:
            return True
        request = _FlushRequest(done=threading.Event())
        self._queue.put(request)
        return request.done.wait(timeout=timeout)

    def shutdown(self, timeout: float = 30.0) -> None:
        """Flush queue and stop writer thread."""
        if self._shutdown:
            return

        # Send sentinel before setting _shutdown so in-flight on_span_end calls
        # can still queue their jobs (they check _shutdown before putting).
        self._queue.put(None)
        self._writer_thread.join(timeout=timeout)
        self._shutdown = True

        # Drain any jobs that arrived after the sentinel (race window between
        # sentinel pickup and thread exit where on_span_end could still queue).
        while True:
            try:
                job = self._queue.get_nowait()
                if isinstance(job, _FlushRequest):
                    job.done.set()
                elif job is not None:
                    self._process_job(job)
            except Empty:
                break

        # Finalize any remaining traces (ones that didn't have root span end yet)
        with self._lock:
            for trace in list(self._traces.values()):
                try:
                    self._finalize_trace(trace)
                except Exception as e:
                    logger.warning("Failed to finalize trace %s: %s", trace.trace_id, e)
            self._traces.clear()

    def _get_or_create_trace(self, trace_id: str, name: str) -> TraceState:
        """Get existing trace or create new one.

        Writes directly to config.path (e.g. .trace/). Previous trace contents
        are cleared on new trace creation — only one trace lives in the directory.
        """
        if trace_id in self._traces:
            return self._traces[trace_id]

        trace_path = self._config.path

        # Clear previous trace contents
        if trace_path.exists():
            for child in trace_path.iterdir():
                try:
                    if child.is_dir():
                        shutil.rmtree(child)
                    else:
                        child.unlink()
                except OSError as e:
                    logger.debug("Failed to clean old trace file %s: %s", child, e)

        trace_path.mkdir(parents=True, exist_ok=True)

        trace = TraceState(
            trace_id=trace_id,
            name=name,
            path=trace_path,
            start_time=datetime.now(UTC),
        )
        self._traces[trace_id] = trace

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

            if isinstance(job, _FlushRequest):
                job.done.set()
                continue

            try:
                self._process_job(job)
            except Exception as e:
                logger.warning("Trace debug write failed for span %s: %s", job.span_id, e)

    def _process_job(self, job: WriteJob) -> None:  # noqa: PLR0914
        """Process a span end job - write all span data."""
        with self._lock:
            trace = self._traces.get(job.trace_id)
            if not trace:
                logger.warning("Trace %s not found for span %s", job.trace_id, job.span_id)
                return

            span_info = trace.spans.get(job.span_id)
            if not span_info:
                logger.warning("Span %s not found in trace %s", job.span_id, job.trace_id)
                return

            span_dir = span_info.path

            # Extract input/output from attributes
            input_content = self._extract_json_attribute(job.attributes, "lmnr.span.input")
            output_content = self._extract_json_attribute(job.attributes, "lmnr.span.output")

            # Create content writer
            content_writer = ContentWriter(self._config)

            # Write input/output
            input_ref = content_writer.write(input_content, span_dir, "input")
            output_ref = content_writer.write(output_content, span_dir, "output")

            # Extract span type and metadata
            span_type = self._extract_span_type(job.attributes)
            llm_info = self._extract_llm_info(job.attributes)
            prefect_info = self._extract_prefect_info(job.attributes)

            # Update span info (span_info already validated above)
            end_time = datetime.fromtimestamp(job.end_time_ns / 1e9, tz=UTC)
            span_info.end_time = end_time
            span_info.duration_ms = int((job.end_time_ns - job.start_time_ns) / 1e6)
            span_info.status = "failed" if job.status_code == "ERROR" else "completed"
            span_info.span_type = span_type
            span_info.llm_info = llm_info
            span_info.prefect_info = prefect_info

            # Extract description and expected_cost from span attributes
            span_info.description = job.attributes.get("description")
            ec = job.attributes.get("expected_cost")
            span_info.expected_cost = float(ec) if ec is not None else None

            # Update trace stats
            if llm_info:
                trace.llm_call_count += 1
                trace.total_tokens += llm_info.get("total_tokens", 0)
                trace.total_cost += llm_info.get("cost", 0.0)
                llm_expected = llm_info.get("expected_cost")
                if llm_expected is not None:
                    trace.total_expected_cost += float(llm_expected)

            # Build span metadata (input_ref and output_ref are now dicts)
            span_meta = self._build_span_metadata_v3(job, input_ref, output_ref, span_type, llm_info, prefect_info)

            # Write span.yaml
            span_yaml_path = span_dir / "span.yaml"
            span_yaml_path.write_text(
                yaml.dump(span_meta, default_flow_style=False, allow_unicode=True, sort_keys=False),
                encoding="utf-8",
            )

            self._write_replay_file(job, span_dir)

            # Write events.yaml (OTel span events including log records from the bridge)
            if job.events:
                events_data = self._format_span_events(job.events)
                events_path = span_dir / "events.yaml"
                events_path.write_text(
                    yaml.dump(events_data, default_flow_style=False, allow_unicode=True),
                    encoding="utf-8",
                )

            # Update index
            self._write_index(trace)

            # Finalize trace when ALL spans are completed (not just root)
            running_spans = [s for s in trace.spans.values() if s.status == "running"]
            if not running_spans:
                self._finalize_trace(trace)
                del self._traces[job.trace_id]

    @staticmethod
    def _extract_json_attribute(attributes: dict[str, Any], key: str) -> Any:
        """Extract and JSON-parse a span attribute, returning raw string on parse failure."""
        raw = attributes.get(key)
        if raw:
            try:
                return json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                return raw
        return None

    @staticmethod
    def _extract_span_type(attributes: dict[str, Any]) -> str:
        """Extract span type from attributes."""
        span_type = attributes.get("lmnr.span.type", "DEFAULT")
        type_map = {
            "LLM": "llm",
            "TOOL": "tool",
            "DEFAULT": "default",
        }
        return type_map.get(span_type, "default")

    @staticmethod
    def _extract_llm_info(attributes: dict[str, Any]) -> dict[str, Any] | None:
        """Extract LLM-specific info from attributes."""
        input_tokens = attributes.get("gen_ai.usage.input_tokens") or attributes.get("gen_ai.usage.prompt_tokens")
        output_tokens = attributes.get("gen_ai.usage.output_tokens") or attributes.get("gen_ai.usage.completion_tokens")

        if input_tokens is None and output_tokens is None:
            return None

        cached_tokens = attributes.get("gen_ai.usage.cache_read_input_tokens") or 0

        return {
            "model": (attributes.get("gen_ai.response.model") or attributes.get("gen_ai.request.model") or attributes.get("gen_ai.request_model")),
            "provider": attributes.get("gen_ai.system"),
            "input_tokens": input_tokens or 0,
            "output_tokens": output_tokens or 0,
            "cached_tokens": int(cached_tokens),
            "total_tokens": (input_tokens or 0) + (output_tokens or 0),
            "cost": attributes.get("gen_ai.usage.cost", 0.0),
            "expected_cost": attributes.get("expected_cost"),
            "purpose": attributes.get("purpose"),
        }

    @staticmethod
    def _extract_prefect_info(attributes: dict[str, Any]) -> dict[str, Any] | None:
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

    _REPLAY_FILENAMES: ClassVar[dict[str, str]] = {
        "conversation": "conversation.yaml",
        "pipeline_task": "task.yaml",
        "pipeline_flow": "flow.yaml",
    }

    def _write_replay_file(self, job: WriteJob, span_dir: Path) -> None:
        """Extract replay.payload from span attributes and write as typed YAML."""
        replay_json = job.attributes.get("replay.payload")
        if not (replay_json and isinstance(replay_json, str)):
            return
        try:
            replay_data = json.loads(replay_json)
            payload_type = replay_data.get("payload_type", "conversation")
            filename = self._REPLAY_FILENAMES.get(payload_type, "replay.yaml")
            (span_dir / filename).write_text(
                yaml.dump(replay_data, default_flow_style=False, sort_keys=False, allow_unicode=True),
                encoding="utf-8",
            )
        except (json.JSONDecodeError, OSError) as e:
            logger.debug("Failed to write replay file for span %s: %s", job.span_id, e)

    _EXCLUDED_ATTRIBUTES: frozenset[str] = frozenset({"lmnr.span.input", "lmnr.span.output", "replay.payload"})

    @staticmethod
    def _build_span_metadata_v3(  # noqa: PLR0917
        job: WriteJob,
        input_ref: dict[str, Any],
        output_ref: dict[str, Any],
        span_type: str,
        llm_info: dict[str, Any] | None,
        prefect_info: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Build span metadata dictionary."""
        start_time = datetime.fromtimestamp(job.start_time_ns / 1e9, tz=UTC)
        end_time = datetime.fromtimestamp(job.end_time_ns / 1e9, tz=UTC)
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

        if prefect_info:
            meta["prefect"] = prefect_info

        if llm_info:
            meta["llm"] = llm_info

        description = job.attributes.get("description")
        if description:
            meta["description"] = description
        expected_cost = job.attributes.get("expected_cost")
        if expected_cost is not None:
            meta["expected_cost"] = float(expected_cost)

        meta["input"] = input_ref
        meta["output"] = output_ref

        if job.status_code != "OK" and job.status_description:
            meta["error"] = {
                "message": job.status_description,
            }

        filtered_attrs = {k: v for k, v in job.attributes.items() if k not in LocalTraceWriter._EXCLUDED_ATTRIBUTES}
        if filtered_attrs:
            meta["attributes"] = filtered_attrs

        return meta

    @staticmethod
    def _format_span_events(events: list[Any]) -> list[dict[str, Any]]:
        """Format span events for YAML output."""
        result: list[dict[str, Any]] = []
        for event in events:
            try:
                event_dict = {
                    "name": event.name,
                    "timestamp": datetime.fromtimestamp(event.timestamp / 1e9, tz=UTC).isoformat(),
                }
                if event.attributes:
                    event_dict["attributes"] = dict(event.attributes)
                result.append(event_dict)
            except Exception as e:
                logger.debug("Failed to serialize event '%s': %s", event.name, e)
                continue
        return result

    def _write_index(self, trace: TraceState) -> None:
        """Write index files: llm_calls.yaml, errors.yaml."""
        sorted_spans = sorted(trace.spans.values(), key=lambda s: s.order)
        self._write_llm_index(trace, sorted_spans)
        self._write_errors_index(trace, sorted_spans)

    @staticmethod
    def _write_llm_index(trace: TraceState, sorted_spans: list[SpanInfo]) -> None:
        """Write llm_calls.yaml - LLM-specific details."""
        llm_calls: list[dict[str, Any]] = []

        for span in sorted_spans:
            if span.llm_info:
                relative_path = span.path.relative_to(trace.path).as_posix() + "/"

                parent_context = ""
                if span.parent_id and span.parent_id in trace.spans:
                    parent_span = trace.spans[span.parent_id]
                    parent_context = f" (in {parent_span.name})"

                llm_entry = {
                    "span_id": span.span_id,
                    "name": span.name + parent_context,
                    "model": span.llm_info.get("model"),
                    "provider": span.llm_info.get("provider"),
                    "input_tokens": span.llm_info.get("input_tokens", 0),
                    "output_tokens": span.llm_info.get("output_tokens", 0),
                    "cached_tokens": span.llm_info.get("cached_tokens", 0),
                    "total_tokens": span.llm_info.get("total_tokens", 0),
                    "cost": span.llm_info.get("cost", 0.0),
                    "expected_cost": span.llm_info.get("expected_cost"),
                    "purpose": span.llm_info.get("purpose"),
                    "duration_ms": span.duration_ms,
                    "status": span.status,
                    "path": relative_path,
                }

                if span.start_time:
                    llm_entry["start_time"] = span.start_time.isoformat()

                llm_calls.append(llm_entry)

        llm_data: dict[str, Any] = {
            "format_version": 3,
            "trace_id": trace.trace_id,
            "llm_call_count": len(llm_calls),
            "total_tokens": trace.total_tokens,
            "total_cost": round(trace.total_cost, 6),
            "total_expected_cost": round(trace.total_expected_cost, 6),
            "calls": llm_calls,
        }

        llm_path = trace.path / "llm_calls.yaml"
        llm_path.write_text(
            yaml.dump(llm_data, default_flow_style=False, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )

    @staticmethod
    def _write_errors_index(trace: TraceState, sorted_spans: list[SpanInfo]) -> None:
        """Write errors.yaml - failed spans only."""
        error_spans: list[dict[str, Any]] = []

        for span in sorted_spans:
            if span.status == "failed":
                relative_path = span.path.relative_to(trace.path).as_posix() + "/"

                error_entry: dict[str, Any] = {
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

                parent_chain: list[str] = []
                current_id = span.parent_id
                while current_id and current_id in trace.spans:
                    parent = trace.spans[current_id]
                    parent_chain.append(parent.name)
                    current_id = parent.parent_id
                if parent_chain:
                    error_entry["parent_chain"] = list(reversed(parent_chain))

                error_spans.append(error_entry)

        if error_spans:
            errors_data: dict[str, Any] = {
                "format_version": 3,
                "trace_id": trace.trace_id,
                "error_count": len(error_spans),
                "errors": error_spans,
            }

            errors_path = trace.path / "errors.yaml"
            errors_path.write_text(
                yaml.dump(errors_data, default_flow_style=False, allow_unicode=True, sort_keys=False),
                encoding="utf-8",
            )

    @staticmethod
    def _detect_wrapper_spans(trace: TraceState) -> set[str]:
        """Detect Prefect wrapper spans that should be merged with their inner spans.

        Detection criteria:
        1. Parent has exactly one child
        2. Names match after stripping hash suffix (e.g., "task-abc123" matches "task")
        3. Parent has no I/O (input type is "none")
        4. Parent has prefect.run.id, child does not
        """
        wrappers: set[str] = set()

        for span_id, span in trace.spans.items():
            if len(span.children) != 1:
                continue

            child_id = span.children[0]
            child = trace.spans.get(child_id)
            if not child:
                continue

            parent_base = re.sub(r"-[a-f0-9]{3,}$", "", span.name)
            child_base = re.sub(r"-[a-f0-9]{3,}$", "", child.name)
            if parent_base != child_base:
                continue

            span_yaml = span.path / "span.yaml"
            if span_yaml.exists():
                try:
                    span_meta = yaml.safe_load(span_yaml.read_text())
                    if span_meta.get("input", {}).get("type") != "none":
                        continue
                except Exception as e:
                    logger.debug("Failed to parse span YAML at %s: %s", span_yaml, e)
                    continue

            if not span.prefect_info:
                continue

            if child.prefect_info:
                child_run_id = child.prefect_info.get("run_id")
                parent_run_id = span.prefect_info.get("run_id")
                if child_run_id != parent_run_id:
                    continue

            wrappers.add(span_id)

        return wrappers

    def _merge_wrapper_spans(self, trace: TraceState) -> None:
        """Merge wrapper spans with their inner spans (virtual merge).

        Modifies the span hierarchy so wrappers are skipped in index output.
        Physical directories remain unchanged.
        """
        if not self._config.merge_wrapper_spans:
            return

        wrappers = self._detect_wrapper_spans(trace)
        if not wrappers:
            return

        logger.debug("Merging %s wrapper spans in trace %s", len(wrappers), trace.trace_id)

        trace.merged_wrapper_ids = wrappers

        for wrapper_id in wrappers:
            wrapper = trace.spans[wrapper_id]
            child_id = wrapper.children[0]
            child = trace.spans[child_id]
            grandparent_id = wrapper.parent_id

            child.parent_id = grandparent_id

            if grandparent_id and grandparent_id in trace.spans:
                grandparent = trace.spans[grandparent_id]
                if wrapper_id in grandparent.children:
                    idx = grandparent.children.index(wrapper_id)
                    grandparent.children[idx] = child_id
            elif trace.root_span_id == wrapper_id:
                trace.root_span_id = child_id

            wrapper.children = []

    def _finalize_trace(self, trace: TraceState) -> None:
        """Finalize a trace - generate summary and indexes."""
        # Merge wrapper spans before generating indexes
        self._merge_wrapper_spans(trace)

        # Final index update
        self._write_index(trace)

        # Generate summary if enabled
        if self._config.generate_summary:
            summary = generate_summary(trace)
            (trace.path / "summary.md").write_text(summary, encoding="utf-8")
            costs = generate_costs(trace)
            if costs:
                (trace.path / "costs.md").write_text(costs, encoding="utf-8")

    @staticmethod
    def _sanitize_name(name: str) -> str:
        """Sanitize name for safe filesystem use.

        Produces a name that:
        - Contains only filesystem-safe characters
        - Is at most _MAX_SPAN_NAME_LENGTH characters
        - Ends with an alphanumeric character (no trailing - or _)
        """
        safe = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name)
        safe = safe.strip(". ")

        # Truncate to max length
        if len(safe) > _MAX_SPAN_NAME_LENGTH:
            safe = safe[:_MAX_SPAN_NAME_LENGTH]

        # Strip trailing non-alphanumeric characters
        safe = safe.rstrip("-_ ")

        return safe or "span"
