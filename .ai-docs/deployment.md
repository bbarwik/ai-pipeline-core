# MODULE: deployment
# CLASSES: DeploymentResult, FlowAction, FlowDirective, PipelineDeployment, RemoteDeployment, ReconstructedEvent, DocumentInput
# DEPENDS: BaseModel, Generic, StrEnum
# PURPOSE: Pipeline deployment utilities for unified, type-safe deployments.
# VERSION: 0.15.1
# AUTO-GENERATED from source code — do not edit. Run: make docs-ai-build

## Imports

```python
from ai_pipeline_core import DeploymentResult, PipelineDeployment, RemoteDeployment
from ai_pipeline_core.deployment import DocumentInput, FlowAction, FlowDirective, ReconstructedEvent, reconstruct_lifecycle_events
```

## Public API

```python
class DeploymentResult(BaseModel):
    """Base class for deployment results."""
    success: bool
    error: str | None = None
    model_config = ConfigDict(frozen=True)


# Enum
class FlowAction(StrEnum):
    """Directive action for dynamic flow control."""
    CONTINUE = 'continue'
    SKIP = 'skip'


@dataclass(frozen=True, slots=True)
class FlowDirective:
    """Flow planning directive returned by plan_next_flow()."""
    action: FlowAction = FlowAction.CONTINUE
    reason: str = ''


class PipelineDeployment(Generic[TOptions, TResult]):
    """Base class for pipeline deployments with three execution modes.

- ``run_cli()``: Database-backed (ClickHouse or filesystem)
- ``run_local()``: In-memory database (ephemeral)
- ``as_prefect_flow()``: auto-configured from settings"""
    name: ClassVar[str]
    options_type: ClassVar[type[FlowOptions]]
    result_type: ClassVar[type[DeploymentResult]]
    pubsub_service_type: ClassVar[str] = ''
    cache_ttl: ClassVar[timedelta | None] = timedelta(hours=24)
    concurrency_limits: ClassVar[Mapping[str, PipelineLimit]] = MappingProxyType({})

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)

        if cls.__name__.startswith("Test"):
            raise TypeError(f"Deployment class name cannot start with 'Test': {cls.__name__}")

        if "name" not in cls.__dict__:
            cls.name = class_name_to_deployment_name(cls.__name__)

        generic_args = extract_generic_params(cls, PipelineDeployment)
        if len(generic_args) < 2:
            raise TypeError(f"{cls.__name__} must specify Generic parameters: class {cls.__name__}(PipelineDeployment[MyOptions, MyResult])")
        options_type, result_type = generic_args[0], generic_args[1]

        cls.options_type = options_type
        cls.result_type = result_type

        # build_result must be implemented (not still abstract from PipelineDeployment)
        build_result_fn = getattr(cls, "build_result", None)
        if build_result_fn is None or getattr(build_result_fn, "__isabstractmethod__", False):
            raise TypeError(f"{cls.__name__} must implement 'build_result' static method")

        if _first_declaring_class(cls, "build_flows") is PipelineDeployment:
            raise TypeError(f"{cls.__name__} must implement build_flows(options) -> Sequence[PipelineFlow]. Decorator-based `flows = [...]` is removed.")

        # Concurrency limits validation
        cls.concurrency_limits = _validate_concurrency_limits(cls.__name__, getattr(cls, "concurrency_limits", MappingProxyType({})))

    @final
    def as_prefect_flow(self) -> Callable[..., Any]:
        """Generate a Prefect flow for production deployment via ``ai-pipeline-deploy`` CLI."""
        return build_prefect_flow(self)

    def build_flows(self, options: TOptions) -> Sequence[PipelineFlow]:
        """Build flow instances for this run."""
        raise NotImplementedError(f"{type(self).__name__}.build_flows() must return a sequence of PipelineFlow.")

    def build_partial_result(self, run_id: str, documents: tuple[Document, ...], options: TOptions) -> TResult:
        """Build a result for partial pipeline runs (--start/--end that don't reach the last step).

        Override this method to customize partial run results. Default delegates to build_result.
        """
        return self.build_result(run_id, documents, options)

    @staticmethod
    @abstractmethod
    def build_result(run_id: str, documents: tuple[Document, ...], options: TOptions) -> TResult:
        """Extract typed result from pipeline documents.

        Called for both full runs and partial runs (--start/--end). For partial runs,
        build_partial_result() delegates here by default — override build_partial_result()
        to customize partial run results.
        """
        ...

    def plan_next_flow(
        self,
        flow_class: type[PipelineFlow],
        plan: Sequence[PipelineFlow],
        output_documents: tuple[Document, ...],
    ) -> FlowDirective:
        """Optionally skip future instances of a flow class."""
        _ = (flow_class, plan, output_documents)
        return FlowDirective()

    @final
    async def run(
        self,
        run_id: str,
        documents: Sequence[Document],
        options: TOptions,
        publisher: ResultPublisher | None = None,
        start_step: int = 1,
        end_step: int | None = None,
        parent_execution_id: UUID | None = None,
        database: Any = None,
    ) -> TResult:
        """Execute flows with resume, per-flow uploads, and step control.

        run_id must match ``[a-zA-Z0-9_-]+``, max 100 chars.
        """
        return await self._run_with_context(
            run_id,
            documents,
            options,
            parent_deployment_task_id=None,
            publisher=publisher,
            start_step=start_step,
            end_step=end_step,
            parent_execution_id=parent_execution_id,
            database=database,
        )

    @final
    def run_cli(
        self,
        initializer: Callable[[TOptions], tuple[str, tuple[Document, ...]]] | None = None,
        cli_mixin: type | None = None,
    ) -> None:
        """Execute pipeline from CLI with positional working_directory and --start/--end flags."""
        run_cli_for_deployment(self, initializer, cli_mixin)

    @final
    def run_local(
        self,
        run_id: str,
        documents: Sequence[Document],
        options: TOptions,
        publisher: ResultPublisher | None = None,
        output_dir: Path | None = None,
    ) -> TResult:
        """Run locally with Prefect test harness and in-memory database.

        Args:
            run_id: Pipeline run identifier.
            documents: Initial input documents.
            options: Flow options.
            publisher: Optional lifecycle event publisher (defaults to _NoopPublisher).
            output_dir: Optional directory for writing result.json.

        Returns:
            Typed deployment result.
        """
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)

        with prefect_test_harness(), disable_run_logger():
            result = asyncio.run(self.run(run_id, documents, options, publisher=publisher, database=MemoryDatabase()))

        if output_dir:
            (output_dir / "result.json").write_text(result.model_dump_json(indent=2))

        return result


class RemoteDeployment(Generic[TOptions, TResult]):
    """Typed client for calling a remote PipelineDeployment via Prefect.

Generic parameters:
    TOptions: FlowOptions subclass for the deployment.
    TResult: DeploymentResult subclass returned by the deployment.

Set ``deployment_class`` to enable inline mode (test/local):
    deployment_class = "module.path:ClassName""""
    name: ClassVar[str]
    options_type: ClassVar[type[FlowOptions]]
    result_type: ClassVar[type[DeploymentResult]]
    deployment_class: ClassVar[str] = ''

    @property
    def deployment_path(self) -> str:
        """Full Prefect deployment path: '{flow_name}/{deployment_name}'."""
        return f"{self.name}/{self.name.replace('-', '_')}"

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)

        # Auto-derive name unless explicitly set in class body
        if "name" not in cls.__dict__:
            cls.name = class_name_to_deployment_name(cls.__name__)

        # Extract Generic params: (TOptions, TResult)
        generic_args = extract_generic_params(cls, RemoteDeployment)
        if len(generic_args) < 2:
            raise TypeError(f"{cls.__name__} must specify 2 Generic parameters: class {cls.__name__}(RemoteDeployment[OptionsType, ResultType])")

        options_type, result_type = generic_args[0], generic_args[1]

        if not isinstance(options_type, type) or not issubclass(options_type, FlowOptions):
            raise TypeError(f"{cls.__name__}: first Generic param must be a FlowOptions subclass, got {options_type}")
        if not isinstance(result_type, type) or not issubclass(result_type, DeploymentResult):
            raise TypeError(f"{cls.__name__}: second Generic param must be a DeploymentResult subclass, got {result_type}")

        cls.options_type = options_type
        cls.result_type = result_type

    @final
    async def run(
        self,
        documents: tuple[Document, ...],
        options: TOptions,
    ) -> TResult:
        """Execute the remote deployment.

        Uses inline mode when the active database backend cannot support remote execution,
        and Prefect remote mode when it can.
        """
        run_id = get_run_id()
        validate_run_id(run_id)
        derived_run_id = _derive_remote_run_id(run_id, documents, options)
        validate_run_id(derived_run_id)

        # Get execution context for DAG linking
        exec_ctx = get_execution_context()
        database = exec_ctx.database if exec_ctx else None
        deployment_id = exec_ctx.deployment_id if exec_ctx else None
        root_deployment_id = exec_ctx.root_deployment_id if exec_ctx else None

        subtask_span_id = uuid7()
        parent_span_id = (exec_ctx.current_span_id or deployment_id) if exec_ctx is not None else deployment_id
        sequence_no = exec_ctx.next_child_sequence(parent_span_id) if exec_ctx is not None and parent_span_id is not None else 0
        deployment_name = exec_ctx.deployment_name if exec_ctx is not None else ""

        # Determine backend mode
        use_inline = database is None
        inline_reason = "no execution database context is active"
        if database is not None and not database.supports_remote:
            use_inline = True
            inline_reason = "the active database backend does not support remote execution"

        publisher = exec_ctx.publisher if exec_ctx else None
        flow_frame = exec_ctx.flow_frame if exec_ctx else None
        flow_step = flow_frame.step if flow_frame is not None else 0
        total_steps = flow_frame.total_steps if flow_frame is not None else 0
        flow_name = flow_frame.name if flow_frame is not None else ""
        parent_deployment_task_id = exec_ctx.parent_deployment_task_id if exec_ctx else None
        input_sha256s = tuple(doc.sha256 for doc in documents)
        task_name = f"remote:{self.name}"
        task_start = time.monotonic()

        async with track_span(
            SpanKind.TASK,
            task_name,
            "",
            sinks=get_sinks(),
            span_id=subtask_span_id,
            parent_span_id=parent_span_id,
            encode_input={"documents": tuple(documents), "options": options},
            db=database,
            input_preview={"deployment": self.name, "document_count": len(documents)},
        ) as span_ctx:
            span_ctx.set_meta(
                deployment_name=deployment_name,
                remote_mode="inline" if use_inline else "prefect",
                sequence_no=sequence_no,
            )
            if publisher is not None and flow_frame is not None:
                try:
                    await publisher.publish_task_started(
                        TaskStartedEvent(
                            run_id=run_id,
                            span_id=str(subtask_span_id),
                            root_deployment_id=str(root_deployment_id or ""),
                            parent_deployment_task_id=str(parent_deployment_task_id) if parent_deployment_task_id else None,
                            flow_name=flow_name,
                            step=flow_step,
                            total_steps=total_steps,
                            status=str(SpanStatus.RUNNING),
                            task_name=task_name,
                            task_class=type(self).__name__,
                            parent_span_id=str(exec_ctx.flow_span_id) if exec_ctx is not None and exec_ctx.flow_span_id else "",
                            input_document_sha256s=input_sha256s,
                        )
                    )
                except (OSError, RuntimeError, ValueError, TypeError) as exc:
                    logger.warning("Remote task started event publish failed for '%s': %s", task_name, exc)
            try:
                if use_inline:
                    logger.warning(
                        "RemoteDeployment '%s' is falling back to inline execution because %s. "
                        "Configure a non-local deployment database to force Prefect remote execution.",
                        self.name,
                        inline_reason,
                    )
                    result = await self._run_inline(
                        derived_run_id,
                        documents,
                        options,
                        root_deployment_id=root_deployment_id or deployment_id or subtask_span_id,
                        parent_deployment_task_id=subtask_span_id,
                        database=database,
                        publisher=publisher,
                        parent_execution_id=exec_ctx.execution_id if exec_ctx else None,
                    )
                else:
                    result = await self._run_remote(
                        derived_run_id,
                        documents,
                        options,
                        root_deployment_id=root_deployment_id or deployment_id or subtask_span_id,
                        parent_deployment_task_id=subtask_span_id,
                        parent_execution_id=exec_ctx.execution_id if exec_ctx else None,
                    )
            except (Exception, asyncio.CancelledError) as exc:
                if publisher is not None and flow_frame is not None:
                    try:
                        await publisher.publish_task_failed(
                            TaskFailedEvent(
                                run_id=run_id,
                                span_id=str(subtask_span_id),
                                root_deployment_id=str(root_deployment_id or ""),
                                parent_deployment_task_id=str(parent_deployment_task_id) if parent_deployment_task_id else None,
                                flow_name=flow_name,
                                step=flow_step,
                                total_steps=total_steps,
                                status=str(SpanStatus.FAILED),
                                task_name=task_name,
                                task_class=type(self).__name__,
                                error_message=str(exc),
                                parent_span_id=str(exec_ctx.flow_span_id) if exec_ctx is not None and exec_ctx.flow_span_id else "",
                                input_document_sha256s=input_sha256s,
                            )
                        )
                    except (OSError, RuntimeError, ValueError, TypeError) as pub_exc:
                        logger.warning("Remote task failed event publish failed for '%s': %s", task_name, pub_exc)
                raise
            if publisher is not None and flow_frame is not None:
                try:
                    await publisher.publish_task_completed(
                        TaskCompletedEvent(
                            run_id=run_id,
                            span_id=str(subtask_span_id),
                            root_deployment_id=str(root_deployment_id or ""),
                            parent_deployment_task_id=str(parent_deployment_task_id) if parent_deployment_task_id else None,
                            flow_name=flow_name,
                            step=flow_step,
                            total_steps=total_steps,
                            status=str(SpanStatus.COMPLETED),
                            task_name=task_name,
                            task_class=type(self).__name__,
                            duration_ms=int((time.monotonic() - task_start) * 1000),
                            parent_span_id=str(exec_ctx.flow_span_id) if exec_ctx is not None and exec_ctx.flow_span_id else "",
                            input_document_sha256s=input_sha256s,
                        )
                    )
                except (OSError, RuntimeError, ValueError, TypeError) as pub_exc:
                    logger.warning("Remote task completed event publish failed for '%s': %s", task_name, pub_exc)
            span_ctx.set_output_preview(result.model_dump(mode="json"))
            span_ctx._set_output_value(result)
            return result


@dataclass(frozen=True, slots=True)
class ReconstructedEvent:
    """A lifecycle event reconstructed from database spans."""
    event_type: EventType
    span_id: str
    timestamp: datetime
    data: dict[str, Any]


class DocumentInput(_InputBase):
    """Document provided to a deployment — inline content or a URL reference."""
    name: str = Field(default='', description="Document filename (e.g. 'task.md'). Auto-derived from URL path if omitted.")  # Document filename (e.g. 'task.md'). Auto-derived from URL path if omitted.
    description: str = Field(default='', description='Human-readable description of this document.')  # Human-readable description of this document.
    summary: str = Field(default='', description='Inline summary of the document content.')  # Inline summary of the document content.
    class_name: str = Field(default='', description='Document type class name. Required when the pipeline accepts multiple input types.')  # Document type class name. Required when the pipeline accepts multiple input types.
    derived_from: tuple[str, ...] = Field(default=(), description='Content provenance: SHA256 hashes of source documents or URIs.')  # Content provenance: SHA256 hashes of source documents or URIs.
    triggered_by: tuple[str, ...] = Field(default=(), description='Causal provenance: SHA256 hashes of triggering documents.')  # Causal provenance: SHA256 hashes of triggering documents.
    attachments: tuple[AttachmentInput, ...] = Field(default=(), description='Secondary content attached to this document.')  # Secondary content attached to this document.
    STRIP_KEYS: ClassVar[frozenset[str]] = frozenset({'id', 'sha256', 'content_sha256', 'size', 'mime_type'})


```

## Functions

```python
async def reconstruct_lifecycle_events(
    reader: DatabaseReader,
    root_deployment_id: UUID,
) -> list[ReconstructedEvent]:
    """Reconstruct lifecycle events from a deployment's span tree.

    Returns events in chronological order. Each event's data dict matches
    the payload shape published by PubSubPublisher (via event_to_payload()).

    Heartbeats are not reconstructed (ephemeral, not stored in spans).
    """
    all_spans = await reader.get_deployment_tree(root_deployment_id)

    lifecycle_spans = [s for s in all_spans if s.kind in _LIFECYCLE_KINDS]
    if not lifecycle_spans:
        return []

    all_doc_shas: set[str] = set()
    for span in lifecycle_spans:
        all_doc_shas.update(span.input_document_shas)
        all_doc_shas.update(span.output_document_shas)

    doc_map: dict[str, DocumentRecord] = {}
    if all_doc_shas:
        doc_map = await reader.get_documents_batch(sorted(all_doc_shas))

    span_by_id: dict[UUID, SpanRecord] = {s.span_id: s for s in lifecycle_spans}
    meta_by_id: dict[UUID, dict[str, Any]] = {s.span_id: _parse_meta(s) for s in lifecycle_spans}

    deployment_span: SpanRecord | None = None
    for span in lifecycle_spans:
        if span.kind == SpanKind.DEPLOYMENT:
            deployment_span = span
            break

    parent_task_id_str: str | None = None
    deployment_span_id_str = ""
    if deployment_span is not None:
        parent_task_id_str = str(deployment_span.parent_span_id) if deployment_span.parent_span_id else None
        deployment_span_id_str = str(deployment_span.span_id)

    events: list[ReconstructedEvent] = []

    for span in lifecycle_spans:
        meta = meta_by_id[span.span_id]

        if span.kind == SpanKind.DEPLOYMENT:
            events.extend(_reconstruct_deployment_events(span, meta, parent_task_id_str))

        elif span.kind == SpanKind.FLOW:
            events.extend(_reconstruct_flow_events(span, meta, parent_task_id_str, deployment_span_id_str, doc_map))

        elif span.kind == SpanKind.TASK:
            flow_span = span_by_id.get(span.parent_span_id) if span.parent_span_id else None
            flow_meta = meta_by_id.get(span.parent_span_id, {}) if span.parent_span_id else {}
            events.extend(_reconstruct_task_events(span, meta, parent_task_id_str, flow_span=flow_span, flow_meta=flow_meta, doc_map=doc_map))

    events.sort(key=_sort_key)
    return events

```

## Examples

**Format starts with base run id** (`tests/deployment/test_remote_deployment.py:769`)

```python
def test_format_starts_with_base_run_id(self):
    """Derived run_id starts with the user's base run_id."""
    doc = AlphaDoc.create_root(name="test.txt", content="hello", reason="test")
    derived = _derive_remote_run_id("my-project", [doc], FlowOptions())
    assert derived.startswith("my-project-")
```

**Deployment result data** (`tests/deployment/test_deployment_base.py:161`)

```python
def test_deployment_result_data(self):
    """Test DeploymentResultData."""
    data = DeploymentResultData(success=True, error=None)
    assert data.success is True
    dumped = data.model_dump()
    assert "success" in dumped
```

**Document input fields have descriptions** (`tests/deployment/test_deploy.py:311`)

```python
def test_document_input_fields_have_descriptions(self):
    """All DocumentInput fields must have a description in JSON schema output."""
    from ai_pipeline_core.deployment._resolve import DocumentInput

    schema = DocumentInput.model_json_schema()
    props = schema["properties"]
    for field_name in ("content", "url", "name", "description", "class_name", "derived_from", "triggered_by", "attachments"):
        assert "description" in props[field_name], f"DocumentInput.{field_name} missing description"
```

**Extracts remote deployment params** (`tests/deployment/test_helpers.py:57`)

```python
def test_extracts_remote_deployment_params(self):
    """Test correct extraction from RemoteDeployment subclass (2 params)."""
    params = extract_generic_params(SampleRemote, RemoteDeployment)
    assert len(params) == 2
    assert params[0] is FlowOptions
    assert params[1] is SampleResult
```

**Two args returned by helper** (`tests/deployment/test_remote_deployment.py:124`)

```python
def test_two_args_returned_by_helper(self):
    class Foo(RemoteDeployment[FlowOptions, SimpleResult]):
        pass

    args = extract_generic_params(Foo, RemoteDeployment)
    assert len(args) == 2
    assert args[0] is FlowOptions
    assert args[1] is SimpleResult
```

**Two params from remote deployment** (`tests/deployment/test_remote_deployment.py:716`)

```python
def test_two_params_from_remote_deployment(self):
    class Foo(RemoteDeployment[FlowOptions, SimpleResult]):
        pass

    result = extract_generic_params(Foo, RemoteDeployment)
    assert len(result) == 2
    assert result[0] is FlowOptions
    assert result[1] is SimpleResult
```

**Accepts deployment result subclass** (`tests/deployment/test_remote_deployment.py:164`)

```python
def test_accepts_deployment_result_subclass(self):
    class Foo(RemoteDeployment[FlowOptions, SimpleResult]):
        pass

    assert Foo.result_type is SimpleResult
```

**Accepts flow options subclass** (`tests/deployment/test_remote_deployment.py:144`)

```python
def test_accepts_flow_options_subclass(self):
    class CustomOpts(FlowOptions):
        budget: float = 10.0

    class Good(RemoteDeployment[CustomOpts, SimpleResult]):
        pass

    assert Good.options_type is CustomOpts
```

**Auto derived** (`tests/deployment/test_remote_deployment.py:191`)

```python
def test_auto_derived(self):
    class AiResearch(RemoteDeployment[FlowOptions, SimpleResult]):
        pass

    assert AiResearch().deployment_path == "ai-research/ai_research"
```


## Error Examples

**Aggregates errors** (`tests/deployment/test_resolve.py:179`)

```python
async def test_aggregates_errors(self):
    inputs = [
        DocumentInput(content="ok", name="ok.txt", class_name="NonExistent"),
        DocumentInput(content="ok", name="ok2.txt", class_name="AlsoNonExistent"),
    ]
    with pytest.raises(ValueError, match="Failed to resolve 2/2"):
        await resolve_document_inputs(inputs, [ResolveDoc])
```

**Resolve rejects derived from on input** (`tests/deployment/test_flow_planning.py:166`)

```python
@pytest.mark.asyncio
async def test_resolve_rejects_derived_from_on_input() -> None:
    """DocumentInput with derived_from raises ValueError."""
    inputs = [DocumentInput(content="hello", name="x.txt", class_name="ResolveInputDoc", derived_from=("SOMESHA256",))]
    with pytest.raises(ValueError, match="cannot set derived_from"):
        await resolve_document_inputs(inputs, [ResolveInputDoc])
```

**Resolve rejects triggered by on input** (`tests/deployment/test_flow_planning.py:174`)

```python
@pytest.mark.asyncio
async def test_resolve_rejects_triggered_by_on_input() -> None:
    """DocumentInput with triggered_by raises ValueError."""
    inputs = [DocumentInput(content="hello", name="x.txt", class_name="ResolveInputDoc", triggered_by=("SOMESHA256",))]
    with pytest.raises(ValueError, match="cannot set derived_from"):
        await resolve_document_inputs(inputs, [ResolveInputDoc])
```

**Ambiguous raises** (`tests/deployment/test_resolve.py:146`)

```python
async def test_ambiguous_raises(self):
    inputs = [DocumentInput(content="data", name="doc.txt")]
    with pytest.raises(ValueError, match="Multiple input types"):
        await resolve_document_inputs(inputs, [ResolveDoc, OtherDoc], start_step_input_types=[ResolveDoc, OtherDoc])
```

**Attachment no name inline raises** (`tests/deployment/test_resolve.py:164`)

```python
async def test_attachment_no_name_inline_raises(self):
    att = AttachmentInput(content="data", name="")
    inputs = [DocumentInput(content="main", name="d.txt", class_name="ResolveDoc", attachments=(att,))]
    with pytest.raises(ValueError, match="must have a name"):
        await resolve_document_inputs(inputs, [ResolveDoc])
```

**Both raises** (`tests/deployment/test_resolve.py:47`)

```python
def test_both_raises(self):
    with pytest.raises(ValueError, match="cannot have both"):
        DocumentInput(url="https://example.com", content="hello", name="d")
```

**Deployment requires build flows override** (`tests/deployment/test_deployment_base.py:705`)

```python
def test_deployment_requires_build_flows_override():
    with pytest.raises(TypeError, match="build_flows"):

        class MissingFlows(PipelineDeployment[_TestOptions, _TestResult]):
            @staticmethod
            def build_result(run_id, documents, options):
                _ = (run_id, documents, options)
                return _TestResult(success=True)
```
