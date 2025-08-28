"""Example demonstrating AI Pipeline Core logging capabilities"""

import asyncio
import time
from typing import List

from ai_pipeline_core.logging_config import setup_logging
from ai_pipeline_core.logging_mixin import LoggerMixin, PrefectLoggerMixin, StructuredLoggerMixin
from prefect import flow, task
from prefect.logging import get_logger, get_run_logger

from ai_pipeline_core.documents import Document

# Initialize logging
setup_logging(level="INFO")


# Example 1: Basic class with logging
class DocumentProcessor(LoggerMixin):
    """Example of basic logging in a document processor"""

    _logger_name = "ai_pipeline_core.documents"

    def process(self, document: Document) -> str:
        """Process a document with comprehensive logging"""
        # Log with context
        self.log_info(
            f"Processing document: {document.name}",
            document_id=document.id,
            mime_type=document.mime_type,
            size_bytes=document.size,
        )

        try:
            # Check document size
            if document.size > 1_000_000:  # 1MB
                self.log_warning(
                    f"Large document detected: {document.name}", size_mb=document.size / 1_024_000
                )

            # Simulate processing
            time.sleep(0.1)
            result = f"Processed content from {document.name}"

            self.log_info(f"Successfully processed: {document.name}")
            return result

        except Exception as e:
            self.log_error(
                f"Failed to process {document.name}",
                exc_info=True,  # Include traceback
                document_id=document.id,
                error_type=type(e).__name__,
            )
            raise


# Example 2: Structured logging with metrics
class LLMService(StructuredLoggerMixin):
    """Example of structured logging with metrics tracking"""

    _logger_name = "ai_pipeline_core.llm"

    async def generate_text(self, model: str, prompt: str, max_tokens: int = 100) -> str:
        """Generate text with detailed metrics logging"""
        # Use context manager for operation timing
        with self.log_operation("llm_generation", model=model, max_tokens=max_tokens):
            # Log structured event
            self.log_event(
                "llm_request_started", model=model, prompt_length=len(prompt), max_tokens=max_tokens
            )

            # Track timing
            start_time = time.perf_counter()

            # Simulate API call
            await asyncio.sleep(0.5)
            response = f"Generated text for prompt: {prompt[:50]}..."
            tokens_used = len(response.split())

            # Log metrics
            duration_ms = (time.perf_counter() - start_time) * 1000
            self.log_metric("llm_latency", duration_ms, "ms", model=model)
            self.log_metric("tokens_generated", tokens_used, "tokens", model=model)

            # Log span for distributed tracing
            self.log_span(
                "llm_api_call", duration_ms, model=model, tokens_used=tokens_used, status="success"
            )

            return response


# Example 3: Prefect flow with orchestration logging
class PipelineOrchestrator(PrefectLoggerMixin):
    """Example of Prefect-specific logging in flows and tasks"""

    _logger_name = "ai_pipeline_core.flow"

    @task(name="validate-documents", log_prints=True)
    async def validate_documents(self, documents: List[Document]) -> List[Document]:
        """Validate documents with task-level logging"""
        logger = get_run_logger()

        # Log task start
        self.log_task_start("validate-documents", {"count": len(documents)})

        valid_docs = []
        for doc in documents:
            logger.info(f"Validating {doc.name}")

            if doc.size > 0 and doc.mime_type:
                valid_docs.append(doc)
                logger.debug(f"✓ Document {doc.name} is valid")
            else:
                logger.warning(f"✗ Document {doc.name} failed validation")

        # Log task completion
        self.log_task_end("validate-documents", "success", 100.0)

        return valid_docs

    @task(name="process-document", retries=2, retry_delay_seconds=5)
    async def process_single_document(
        self, document: Document, processor: DocumentProcessor
    ) -> str:
        """Process a single document with retry handling"""
        logger = get_run_logger()

        try:
            logger.info(f"Processing {document.name}")
            result = processor.process(document)
            return result

        except Exception as e:
            # Log retry attempt
            self.log_retry("process_document", 1, 2, str(e))
            raise

    @flow(name="document-pipeline", log_prints=True)
    async def run_pipeline(self, documents: List[Document]) -> dict:
        """Main pipeline flow with comprehensive logging"""
        logger = get_run_logger()

        # Log flow start
        self.log_flow_start("document-pipeline", {"document_count": len(documents)})
        pipeline_start = time.perf_counter()

        try:
            # Checkpoint: Validation
            self.log_checkpoint("validation", status="starting")
            valid_docs = await self.validate_documents(documents)
            self.log_checkpoint(
                "validation",
                status="complete",
                valid=len(valid_docs),
                invalid=len(documents) - len(valid_docs),
            )

            # Checkpoint: Processing
            self.log_checkpoint("processing", status="starting")

            processor = DocumentProcessor()
            results = []

            for doc in valid_docs:
                try:
                    result = await self.process_single_document(doc, processor)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Failed to process {doc.name}: {e}")
                    results.append(None)

            self.log_checkpoint(
                "processing",
                status="complete",
                successful=len([r for r in results if r]),
                failed=len([r for r in results if not r]),
            )

            # Calculate and log metrics
            duration_ms = (time.perf_counter() - pipeline_start) * 1000
            success_rate = len([r for r in results if r]) / len(documents) * 100

            self.log_metric("pipeline_duration", duration_ms, "ms")
            self.log_metric("pipeline_success_rate", success_rate, "%")

            # Log flow completion
            self.log_flow_end("document-pipeline", "success", duration_ms)

            return {
                "processed": len([r for r in results if r]),
                "failed": len([r for r in results if not r]),
                "duration_ms": duration_ms,
                "success_rate": success_rate,
            }

        except Exception as e:
            duration_ms = (time.perf_counter() - pipeline_start) * 1000
            self.log_flow_end("document-pipeline", "failure", duration_ms)
            logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
            raise


# Example 4: Module-level logging (outside flow/task context)
def demonstrate_module_logging():
    """Example of logging outside Prefect context"""
    # Get module logger
    logger = get_logger("ai_pipeline_core.example")

    logger.info("Starting module-level operation")

    # Process with context
    logger.info(
        "Processing batch", extra={"batch_id": "batch_001", "size": 100, "timestamp": time.time()}
    )

    # Log with different levels
    logger.debug("Debug information - only visible when DEBUG level set")
    logger.warning("Warning: Resource usage high", extra={"cpu": 85, "memory": 72})

    # Error with traceback
    try:
        raise ValueError("Example error")
    except ValueError:
        logger.error("Error in module operation", exc_info=True)


# Example 5: Advanced patterns
class AdvancedLoggingExamples:
    """Demonstrate advanced logging patterns"""

    @staticmethod
    def conditional_debug_logging():
        """Only log expensive debug info when needed"""
        logger = get_logger("ai_pipeline_core.advanced")

        # Check if debug is enabled before expensive operation
        if logger.isEnabledFor(logging.DEBUG):
            expensive_debug_info = calculate_debug_state()  # Only called if DEBUG
            logger.debug(f"Debug state: {expensive_debug_info}")

    @staticmethod
    def log_with_sampling(sample_rate: float = 0.01):
        """Sample high-frequency logs to reduce volume"""
        import random

        logger = get_logger("ai_pipeline_core.sampling")

        for i in range(10000):
            # Only log 1% of events
            if random.random() < sample_rate:
                logger.debug(f"[SAMPLED] Processing item {i}")

    @staticmethod
    def aggregate_before_logging():
        """Aggregate data before logging to reduce volume"""
        from collections import Counter

        logger = get_logger("ai_pipeline_core.aggregation")

        # Collect events
        events = Counter()
        for i in range(1000):
            event_type = f"type_{i % 10}"
            events[event_type] += 1

        # Log aggregated results instead of individual events
        logger.info(
            "Event summary",
            extra={
                "event_counts": dict(events),
                "total_events": sum(events.values()),
                "unique_types": len(events),
            },
        )


async def main():
    """Run all examples"""
    logger = get_logger("ai_pipeline_core.example")

    logger.info("=" * 60)
    logger.info("AI Pipeline Core Logging Examples")
    logger.info("=" * 60)

    # Example 1: Basic document processing
    logger.info("\n1. Basic Document Processing:")
    processor = DocumentProcessor()
    doc = Document(name="example.pdf", content=b"PDF content here")
    result = processor.process(doc)
    logger.info(f"   Result: {result}")

    # Example 2: LLM service with metrics
    logger.info("\n2. LLM Service with Metrics:")
    llm_service = LLMService()
    response = await llm_service.generate_text(
        model="gpt-4", prompt="Write a haiku about logging", max_tokens=50
    )
    logger.info(f"   Generated: {response}")

    # Example 3: Prefect pipeline
    logger.info("\n3. Prefect Pipeline Orchestration:")
    orchestrator = PipelineOrchestrator()
    documents = [Document(name=f"doc{i}.txt", content=f"Content {i}".encode()) for i in range(5)]
    pipeline_result = await orchestrator.run_pipeline(documents)
    logger.info(f"   Pipeline result: {pipeline_result}")

    # Example 4: Module-level logging
    logger.info("\n4. Module-level Logging:")
    demonstrate_module_logging()

    # Example 5: Advanced patterns
    logger.info("\n5. Advanced Logging Patterns:")
    AdvancedLoggingExamples.aggregate_before_logging()
    logger.info("   (Check logs for aggregated event summary)")

    logger.info("\n" + "=" * 60)
    logger.info("Examples completed! Check the logs for detailed output.")
    logger.info("=" * 60)


# Helper function for conditional debug example
def calculate_debug_state():
    """Expensive debug calculation"""
    import hashlib

    state = {}
    for i in range(100):
        state[f"item_{i}"] = hashlib.md5(str(i).encode()).hexdigest()
    return state


if __name__ == "__main__":
    import logging

    # Run the examples
    asyncio.run(main())
