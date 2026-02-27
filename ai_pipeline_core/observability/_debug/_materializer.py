"""TraceMaterializer: single code path for .trace/ directory generation.

Receives SpanData objects and produces the hierarchical directory structure
with span.yaml, input/output, replay files, indexes, summary, and costs.
"""

import json
import re
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from ai_pipeline_core.logging import get_pipeline_logger
from ai_pipeline_core.observability._span_data import _CONTENT_ATTRS, SpanData

from ._config import REPLAY_PAYLOAD_TO_FILENAME, SpanInfo, TraceDebugConfig, TraceState
from ._content import ContentWriter
from ._summary import generate_costs, generate_summary

logger = get_pipeline_logger(__name__)

_MAX_SPAN_NAME_LENGTH = 20


class TraceMaterializer:
    """Materializes SpanData into a hierarchical .trace/ directory structure.

    Used by both FilesystemBackend (live spans during execution) and
    the download path (spans reconstructed from ClickHouse rows).

    Thread safety: NOT thread-safe. The FilesystemBackend wraps this in a
    background thread with a queue for serialized access.
    """

    def __init__(self, config: TraceDebugConfig) -> None:
        self._config = config
        self._traces: dict[str, TraceState] = {}
        self._initial_clean_done = False

        config.path.mkdir(parents=True, exist_ok=True)

    def on_span_start(self, span_data: SpanData) -> None:
        """Create directories for a starting span (fast mkdir).

        Uses only identity and ordering fields from SpanData.
        """
        trace = self._get_or_create_trace(span_data.trace_id, span_data.name)

        if span_data.parent_span_id and span_data.parent_span_id in trace.spans:
            parent_info = trace.spans[span_data.parent_span_id]
            parent_path = parent_info.path
            depth = parent_info.depth + 1
        elif span_data.parent_span_id:
            logger.warning("Span %s has unknown parent %s, placing at trace root", span_data.span_id, span_data.parent_span_id)
            parent_path = trace.path
            depth = 0
        else:
            parent_path = trace.path
            depth = 0

        trace.span_counter += 1
        safe_name = _sanitize_name(span_data.name)
        counter = trace.span_counter
        width = 3 if counter < 1000 else len(str(counter))
        dir_name = f"{counter:0{width}d}_{safe_name}"

        span_dir = parent_path / dir_name
        span_dir.mkdir(parents=True, exist_ok=True)

        now = datetime.now(UTC)
        span_info = SpanInfo(
            span_id=span_data.span_id,
            parent_id=span_data.parent_span_id,
            name=span_data.name,
            span_type="default",
            status="running",
            start_time=now,
            path=span_dir,
            depth=depth,
            order=trace.span_counter,
        )
        trace.spans[span_data.span_id] = span_info

        if span_data.parent_span_id is None:
            trace.root_span_id = span_data.span_id

        if span_data.parent_span_id and span_data.parent_span_id in trace.spans:
            trace.spans[span_data.parent_span_id].children.append(span_data.span_id)

    def add_span(self, span_data: SpanData) -> None:
        """Process a completed span -- write all span data to disk.

        If on_span_start was not called for this span (e.g. download path),
        the directory is created here.
        """
        trace = self._get_or_create_trace(span_data.trace_id, span_data.name)

        # If span directory doesn't exist yet (download path), create it
        span_info = trace.spans.get(span_data.span_id)
        if span_info is None:
            self.on_span_start(span_data)
            span_info = trace.spans[span_data.span_id]

        _write_span_files(span_data, span_info, trace, self._config)

        # Update indexes
        self._write_index(trace)

        # Finalize trace when all spans are completed
        if not any(s.status == "running" for s in trace.spans.values()):
            self._finalize_trace(trace)
            del self._traces[span_data.trace_id]

    def record_filtered_llm_metrics(self, span_data: SpanData) -> None:
        """Record cost/token metrics from a filtered LLM span for trace totals.

        Accumulates metrics from ALL filtered LLM spans so that summary.md
        and costs.md reflect actual total costs including regular LLM calls.
        """
        trace = self._traces.get(span_data.trace_id)
        if not trace:
            return
        if span_data.cost > 0 or span_data.tokens_input > 0:
            trace.llm_call_count += 1
            trace.total_tokens += span_data.tokens_input + span_data.tokens_output
            trace.total_cost += span_data.cost

    def finalize_all(self) -> None:
        """Finalize any remaining traces (shutdown)."""
        for trace in list(self._traces.values()):
            try:
                self._finalize_trace(trace)
            except (OSError, ValueError, TypeError, KeyError, RuntimeError) as e:
                logger.warning("Failed to finalize trace %s: %s", trace.trace_id, e)
        self._traces.clear()

    def _get_or_create_trace(self, trace_id: str, name: str) -> TraceState:
        """Get existing trace or create new one.

        Cleans the output directory only on the first trace creation to avoid
        wiping data from other trace_ids in the same download session.
        """
        if trace_id in self._traces:
            return self._traces[trace_id]

        trace_path = self._config.path
        if not self._initial_clean_done:
            self._initial_clean_done = True
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

    @staticmethod
    def _write_index(trace: TraceState) -> None:
        """Write index files: llm_calls.yaml, errors.yaml."""
        sorted_spans = sorted(trace.spans.values(), key=lambda s: s.order)
        _write_llm_index(trace, sorted_spans)
        _write_errors_index(trace, sorted_spans)

    def _finalize_trace(self, trace: TraceState) -> None:
        """Finalize trace -- merge wrappers, generate summary and indexes."""
        if self._config.merge_wrapper_spans:
            _merge_wrapper_spans(trace)

        self._write_index(trace)

        if self._config.generate_summary:
            summary = generate_summary(trace)
            (trace.path / "summary.md").write_text(summary, encoding="utf-8")
            costs = generate_costs(trace)
            if costs:
                (trace.path / "costs.md").write_text(costs, encoding="utf-8")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_span_files(span_data: SpanData, span_info: SpanInfo, trace: TraceState, config: TraceDebugConfig) -> None:
    """Write all span data files (span.yaml, input/output, replay, events) to disk."""
    span_dir = span_info.path

    content_writer = ContentWriter(config)
    input_ref = content_writer.write(_parse_json_safe(span_data.input_json), span_dir, "input")
    output_ref = content_writer.write(_parse_json_safe(span_data.output_json), span_dir, "output")

    span_type = _extract_span_type(span_data.attributes)
    llm_info = _extract_llm_info(span_data.attributes, span_data)
    prefect_info = _extract_prefect_info(span_data.attributes)

    # Update span info
    span_info.end_time = span_data.end_time
    span_info.duration_ms = span_data.duration_ms
    span_info.status = span_data.status
    span_info.span_type = span_type
    span_info.llm_info = llm_info
    span_info.prefect_info = prefect_info
    span_info.description = span_data.attributes.get("description")
    ec = span_data.attributes.get("expected_cost")
    span_info.expected_cost = float(ec) if ec is not None else None

    # Update trace stats
    if llm_info:
        trace.llm_call_count += 1
        trace.total_tokens += llm_info.get("total_tokens", 0)
        trace.total_cost += llm_info.get("cost", 0.0)
        llm_expected = llm_info.get("expected_cost")
        if llm_expected is not None:
            trace.total_expected_cost += float(llm_expected)

    # Write span.yaml
    span_meta = _build_span_metadata(span_data, input_ref, output_ref, span_type=span_type, llm_info=llm_info, prefect_info=prefect_info)
    (span_dir / "span.yaml").write_text(
        yaml.dump(span_meta, default_flow_style=False, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    _write_replay_file(span_data, span_dir)

    if span_data.events:
        events_data = _format_events(span_data.events)
        (span_dir / "events.yaml").write_text(
            yaml.dump(events_data, default_flow_style=False, allow_unicode=True),
            encoding="utf-8",
        )


def _sanitize_name(name: str) -> str:
    """Sanitize name for safe filesystem use."""
    safe = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name)
    safe = safe.strip(". ")
    if len(safe) > _MAX_SPAN_NAME_LENGTH:
        safe = safe[:_MAX_SPAN_NAME_LENGTH]
    safe = safe.rstrip("-_ ")
    return safe or "span"


def _parse_json_safe(text: str) -> Any:
    """Parse JSON text, returning None for empty and raw string for invalid JSON."""
    if not text:
        return None
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return text


def _extract_span_type(attributes: dict[str, Any]) -> str:
    """Extract display span type from attributes."""
    span_type = attributes.get("lmnr.span.type", "DEFAULT")
    type_map = {"LLM": "llm", "TOOL": "tool", "DEFAULT": "default"}
    return type_map.get(str(span_type), "default")


def _extract_llm_info(attributes: dict[str, Any], span_data: SpanData) -> dict[str, Any] | None:
    """Extract LLM-specific info from attributes and span data."""
    input_tokens = attributes.get("gen_ai.usage.input_tokens") or attributes.get("gen_ai.usage.prompt_tokens")
    output_tokens = attributes.get("gen_ai.usage.output_tokens") or attributes.get("gen_ai.usage.completion_tokens")

    # Fall back to SpanData fields if attributes don't have tokens
    if input_tokens is None and output_tokens is None:
        if span_data.tokens_input or span_data.tokens_output:
            input_tokens = span_data.tokens_input
            output_tokens = span_data.tokens_output
        else:
            return None

    cached_tokens = attributes.get("gen_ai.usage.cache_read_input_tokens") or span_data.tokens_cached or 0
    cost = attributes.get("gen_ai.usage.cost", span_data.cost) or 0.0

    return {
        "model": (
            attributes.get("gen_ai.response.model") or attributes.get("gen_ai.request.model") or attributes.get("gen_ai.request_model") or span_data.llm_model
        ),
        "provider": attributes.get("gen_ai.system"),
        "input_tokens": input_tokens or 0,
        "output_tokens": output_tokens or 0,
        "cached_tokens": int(cached_tokens),
        "total_tokens": (input_tokens or 0) + (output_tokens or 0),
        "cost": cost,
        "expected_cost": attributes.get("expected_cost"),
        "purpose": attributes.get("purpose"),
    }


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


def _build_span_metadata(
    span_data: SpanData,
    input_ref: dict[str, Any],
    output_ref: dict[str, Any],
    *,
    span_type: str,
    llm_info: dict[str, Any] | None,
    prefect_info: dict[str, Any] | None,
) -> dict[str, Any]:
    """Build span metadata dictionary for span.yaml."""
    meta: dict[str, Any] = {
        "span_id": span_data.span_id,
        "trace_id": span_data.trace_id,
        "parent_id": span_data.parent_span_id,
        "name": span_data.name,
        "type": span_type,
        "timing": {
            "start": span_data.start_time.isoformat(),
            "end": span_data.end_time.isoformat(),
            "duration_ms": span_data.duration_ms,
        },
        "status": span_data.status,
    }

    if prefect_info:
        meta["prefect"] = prefect_info
    if llm_info:
        meta["llm"] = llm_info

    description = span_data.attributes.get("description")
    if description:
        meta["description"] = description
    expected_cost = span_data.attributes.get("expected_cost")
    if expected_cost is not None:
        meta["expected_cost"] = float(expected_cost)

    meta["input"] = input_ref
    meta["output"] = output_ref

    if span_data.error_message:
        meta["error"] = {"message": span_data.error_message}

    filtered_attrs = {k: v for k, v in span_data.attributes.items() if k not in _CONTENT_ATTRS}
    if filtered_attrs:
        meta["attributes"] = filtered_attrs

    return meta


def _write_replay_file(span_data: SpanData, span_dir: Path) -> None:
    """Write replay.payload as a typed YAML file."""
    if not span_data.replay_payload:
        return
    try:
        replay_data = json.loads(span_data.replay_payload)
        payload_type = replay_data.get("payload_type", "conversation")
        filename = REPLAY_PAYLOAD_TO_FILENAME.get(payload_type, "replay.yaml")
        (span_dir / filename).write_text(
            yaml.dump(replay_data, default_flow_style=False, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
    except (json.JSONDecodeError, OSError) as e:
        logger.debug("Failed to write replay file for span %s: %s", span_data.span_id, e)


def _format_events(events: tuple[dict[str, Any], ...]) -> list[dict[str, Any]]:
    """Format span events for YAML output."""
    result: list[dict[str, Any]] = []
    for event in events:
        event_dict: dict[str, Any] = {"name": event.get("name", "")}
        ts = event.get("timestamp")
        if ts is not None:
            if isinstance(ts, (int, float)):
                event_dict["timestamp"] = datetime.fromtimestamp(ts / 1e9, tz=UTC).isoformat()
            else:
                event_dict["timestamp"] = str(ts)
        attrs = event.get("attributes")
        if attrs:
            event_dict["attributes"] = attrs
        result.append(event_dict)
    return result


def _write_llm_index(trace: TraceState, sorted_spans: list[SpanInfo]) -> None:
    """Write llm_calls.yaml."""
    llm_calls: list[dict[str, Any]] = []
    for span in sorted_spans:
        if span.llm_info:
            relative_path = span.path.relative_to(trace.path).as_posix() + "/"
            parent_context = ""
            if span.parent_id and span.parent_id in trace.spans:
                parent_span = trace.spans[span.parent_id]
                parent_context = f" (in {parent_span.name})"

            llm_entry: dict[str, Any] = {
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
    (trace.path / "llm_calls.yaml").write_text(
        yaml.dump(llm_data, default_flow_style=False, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def _write_errors_index(trace: TraceState, sorted_spans: list[SpanInfo]) -> None:
    """Write errors.yaml for failed spans."""
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
        (trace.path / "errors.yaml").write_text(
            yaml.dump(errors_data, default_flow_style=False, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )


# ---------------------------------------------------------------------------
# Wrapper span merging
# ---------------------------------------------------------------------------


def _detect_wrapper_spans(trace: TraceState) -> set[str]:
    """Detect Prefect wrapper spans that should be merged with their inner spans.

    Detection criteria:
    1. Parent has exactly one child
    2. Names match after stripping hash suffix
    3. Parent has no I/O (input type is "none")
    4. Parent has prefect.run.id, child does not have a different run_id
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
            except (OSError, yaml.YAMLError, TypeError, ValueError) as e:
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


def _merge_wrapper_spans(trace: TraceState) -> None:
    """Merge wrapper spans with their inner spans (virtual merge)."""
    wrappers = _detect_wrapper_spans(trace)
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
