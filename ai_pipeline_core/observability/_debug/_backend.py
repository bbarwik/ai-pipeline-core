"""FilesystemBackend: queue + background thread wrapper around TraceMaterializer.

Implements SpanBackend protocol. All filesystem I/O runs on a dedicated thread.
on_span_start runs synchronously (fast mkdir), on_span_end queues for background write.
"""

import atexit
import threading
from queue import Empty, Queue

from ai_pipeline_core.logging import get_pipeline_logger
from ai_pipeline_core.observability._span_data import SpanData

from ._config import TraceDebugConfig
from ._materializer import TraceMaterializer

logger = get_pipeline_logger(__name__)


class FilesystemBackend:
    """SpanBackend that writes trace data to local filesystem via background thread.

    on_span_start is dispatched synchronously to the materializer under a lock
    so span directories exist before child spans start.

    on_span_end is queued and processed by the background writer thread.
    """

    def __init__(self, config: TraceDebugConfig) -> None:
        self._materializer = TraceMaterializer(config)
        self._verbose = config.verbose
        self._queue: Queue[SpanData | _FlushRequest | None] = Queue()
        self._lock = threading.Lock()
        self._shutdown_flag = False

        self._writer_thread = threading.Thread(
            target=self._writer_loop,
            name="trace-debug-writer",
            daemon=True,
        )
        self._writer_thread.start()
        atexit.register(self.shutdown)

    def on_span_start(self, span_data: SpanData) -> None:
        """Create span directories synchronously (fast mkdir)."""
        with self._lock:
            self._materializer.on_span_start(span_data)

    def on_span_end(self, span_data: SpanData) -> None:
        """Queue span for background filesystem write."""
        if self._shutdown_flag:
            return

        # LLM spans (filtered by processor, span_order=0) only contribute metrics
        if span_data.span_order == 0:
            with self._lock:
                self._materializer.record_filtered_llm_metrics(span_data)
            return

        self._queue.put(span_data)

    def shutdown(self, timeout: float = 30.0) -> None:
        """Flush queue and stop writer thread."""
        if self._shutdown_flag:
            return

        self._queue.put(None)
        self._writer_thread.join(timeout=timeout)
        self._shutdown_flag = True

        # Drain remaining items
        while True:
            try:
                item = self._queue.get_nowait()
                if isinstance(item, _FlushRequest):
                    item.done.set()
                elif item is not None:
                    with self._lock:
                        self._materializer.add_span(item)
            except Empty:
                break

        with self._lock:
            self._materializer.finalize_all()

    def flush(self, timeout: float = 30.0) -> bool:
        """Block until all queued spans are processed."""
        if self._shutdown_flag:
            return True
        request = _FlushRequest()
        self._queue.put(request)
        return request.done.wait(timeout=timeout)

    def _writer_loop(self) -> None:
        """Background thread loop for processing write jobs."""
        while True:
            try:
                item = self._queue.get(timeout=1.0)
            except Empty:
                continue

            if item is None:
                break

            if isinstance(item, _FlushRequest):
                item.done.set()
                continue

            try:
                with self._lock:
                    self._materializer.add_span(item)
            except (OSError, ValueError, TypeError, KeyError, RuntimeError) as e:
                logger.warning("Trace debug write failed for span %s: %s", item.span_id, e)


class _FlushRequest:
    """Sentinel queued to force the writer to drain all preceding jobs."""

    __slots__ = ("done",)

    def __init__(self) -> None:
        self.done = threading.Event()
