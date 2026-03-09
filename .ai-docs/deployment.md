# MODULE: deployment
# CLASSES: DeploymentResult, FlowAction, FlowDirective, PipelineDeployment, RemoteDeployment, DocumentInput
# DEPENDS: BaseModel, Generic, StrEnum
# PURPOSE: Pipeline deployment utilities for unified, type-safe deployments.
# VERSION: 0.14.0
# AUTO-GENERATED from source code — do not edit. Run: make docs-ai-build

## Imports

```python
from ai_pipeline_core import DeploymentResult, PipelineDeployment, RemoteDeployment
from ai_pipeline_core.deployment import DocumentInput, FlowAction, FlowDirective
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
        from ._prefect import build_prefect_flow

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
        database: Database | None = None,
    ) -> TResult:
        """Execute flows with resume, per-flow uploads, and step control.

        run_id must match ``[a-zA-Z0-9_-]+``, max 100 chars.
        """
        deployment_node_id = uuid4()
        root_deployment_id = deployment_node_id
        return await self._run_with_context(
            run_id,
            documents,
            options,
            deployment_node_id=deployment_node_id,
            root_deployment_id=root_deployment_id,
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
        from ._cli import run_cli_for_deployment

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
            result = asyncio.run(self.run(run_id, documents, options, publisher=publisher))

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

        # Pre-allocate child deployment ID
        child_deployment_id = uuid4()
        subtask_node_id = uuid4()

        # Create subtask node in caller's DAG
        if exec_ctx is not None and database is not None and deployment_id is not None and root_deployment_id is not None:
            current_node_id = exec_ctx.current_node_id
            sequence_no = exec_ctx.next_child_sequence(current_node_id) if current_node_id else 0
            subtask_node = ExecutionNode(
                node_id=subtask_node_id,
                node_kind=NodeKind.TASK,
                deployment_id=deployment_id,
                root_deployment_id=root_deployment_id,
                parent_node_id=current_node_id or deployment_id,
                run_id=run_id,
                run_scope=exec_ctx.run_scope if exec_ctx else RunScope(run_id),
                deployment_name=exec_ctx.deployment_name if exec_ctx else "",
                name=f"remote:{self.name}",
                sequence_no=sequence_no,
                task_class=type(self).__name__,
                status=NodeStatus.RUNNING,
                remote_child_deployment_id=child_deployment_id,
                input_document_shas=tuple(d.sha256 for d in documents),
            )
            try:
                await database.insert_node(subtask_node)
            except Exception as exc:
                logger.warning("Failed to insert remote subtask node: %s", exc)

        # Determine backend mode
        use_inline = database is None
        inline_reason = "no execution database context is active"
        if database is not None and not database.supports_remote:
            use_inline = True
            inline_reason = "the active database backend does not support remote execution"

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
                    child_deployment_id=child_deployment_id,
                    root_deployment_id=root_deployment_id or child_deployment_id,
                    parent_deployment_task_id=subtask_node_id,
                    database=cast(Database, database) if database is not None else None,
                    publisher=exec_ctx.publisher if exec_ctx else None,
                    parent_execution_id=exec_ctx.execution_id if exec_ctx else None,
                )
            else:
                result = await self._run_remote(
                    derived_run_id,
                    documents,
                    options,
                    child_deployment_id=child_deployment_id,
                    root_deployment_id=root_deployment_id or child_deployment_id,
                    parent_deployment_task_id=subtask_node_id,
                    parent_execution_id=exec_ctx.execution_id if exec_ctx else None,
                )

            # Update subtask to COMPLETED
            if database is not None:
                try:
                    await database.update_node(
                        subtask_node_id,
                        status=NodeStatus.COMPLETED,
                        ended_at=datetime.now(UTC),
                    )
                except Exception as exc:
                    logger.warning("Failed to update remote subtask node: %s", exc)

            return result

        except (Exception, asyncio.CancelledError) as exc:
            # Update subtask to FAILED
            if database is not None:
                try:
                    await database.update_node(
                        subtask_node_id,
                        status=NodeStatus.FAILED,
                        ended_at=datetime.now(UTC),
                        error_type=type(exc).__name__,
                        error_message=str(exc),
                    )
                except Exception as update_exc:
                    logger.warning("Failed to update remote subtask node on failure: %s", update_exc)
            raise


class DocumentInput(_InputBase):
    """Document provided to a deployment — inline content or a URL reference."""
    name: str = Field(default='', description="Document filename (e.g. 'task.md'). Auto-derived from URL path if omitted.")  # Document filename (e.g. 'task.md'). Auto-derived from URL path if omitted.
    description: str = Field(default='', description='Human-readable description of this document.')  # Human-readable description of this document.
    class_name: str = Field(default='', description='Document type class name. Required when the pipeline accepts multiple input types.')  # Document type class name. Required when the pipeline accepts multiple input types.
    derived_from: tuple[str, ...] = Field(default=(), description='Content provenance: SHA256 hashes of source documents or URIs.')  # Content provenance: SHA256 hashes of source documents or URIs.
    triggered_by: tuple[str, ...] = Field(default=(), description='Causal provenance: SHA256 hashes of triggering documents.')  # Causal provenance: SHA256 hashes of triggering documents.
    attachments: tuple[AttachmentInput, ...] = Field(default=(), description='Secondary content attached to this document.')  # Secondary content attached to this document.
    STRIP_KEYS: ClassVar[frozenset[str]] = frozenset({'id', 'sha256', 'content_sha256', 'size', 'mime_type'})


```

## Examples

**Format starts with base run id** (`tests/deployment/test_remote_deployment.py:721`)

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

**Document input fields have descriptions** (`tests/deployment/test_deploy.py:313`)

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

**Two args returned by helper** (`tests/deployment/test_remote_deployment.py:95`)

```python
def test_two_args_returned_by_helper(self):
    class Foo(RemoteDeployment[FlowOptions, SimpleResult]):
        pass

    args = extract_generic_params(Foo, RemoteDeployment)
    assert len(args) == 2
    assert args[0] is FlowOptions
    assert args[1] is SimpleResult
```

**Two params from remote deployment** (`tests/deployment/test_remote_deployment.py:668`)

```python
def test_two_params_from_remote_deployment(self):
    class Foo(RemoteDeployment[FlowOptions, SimpleResult]):
        pass

    result = extract_generic_params(Foo, RemoteDeployment)
    assert len(result) == 2
    assert result[0] is FlowOptions
    assert result[1] is SimpleResult
```

**Accepts deployment result subclass** (`tests/deployment/test_remote_deployment.py:135`)

```python
def test_accepts_deployment_result_subclass(self):
    class Foo(RemoteDeployment[FlowOptions, SimpleResult]):
        pass

    assert Foo.result_type is SimpleResult
```

**Accepts flow options subclass** (`tests/deployment/test_remote_deployment.py:115`)

```python
def test_accepts_flow_options_subclass(self):
    class CustomOpts(FlowOptions):
        budget: float = 10.0

    class Good(RemoteDeployment[CustomOpts, SimpleResult]):
        pass

    assert Good.options_type is CustomOpts
```

**Auto derived** (`tests/deployment/test_remote_deployment.py:162`)

```python
def test_auto_derived(self):
    class AiResearch(RemoteDeployment[FlowOptions, SimpleResult]):
        pass

    assert AiResearch().deployment_path == "ai-research/ai_research"
```


## Error Examples

**Aggregates errors** (`tests/deployment/test_resolve.py:181`)

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

**Ambiguous raises** (`tests/deployment/test_resolve.py:148`)

```python
async def test_ambiguous_raises(self):
    inputs = [DocumentInput(content="data", name="doc.txt")]
    with pytest.raises(ValueError, match="Multiple input types"):
        await resolve_document_inputs(inputs, [ResolveDoc, OtherDoc], start_step_input_types=[ResolveDoc, OtherDoc])
```

**Attachment no name inline raises** (`tests/deployment/test_resolve.py:166`)

```python
async def test_attachment_no_name_inline_raises(self):
    att = AttachmentInput(content="data", name="")
    inputs = [DocumentInput(content="main", name="d.txt", class_name="ResolveDoc", attachments=(att,))]
    with pytest.raises(ValueError, match="must have a name"):
        await resolve_document_inputs(inputs, [ResolveDoc])
```

**Both raises** (`tests/deployment/test_resolve.py:49`)

```python
def test_both_raises(self):
    with pytest.raises(ValueError, match="cannot have both"):
        DocumentInput(url="https://example.com", content="hello", name="d")
```

**Deployment requires build flows override** (`tests/deployment/test_deployment_base.py:707`)

```python
def test_deployment_requires_build_flows_override():
    with pytest.raises(TypeError, match="build_flows"):

        class MissingFlows(PipelineDeployment[_TestOptions, _TestResult]):
            @staticmethod
            def build_result(run_id, documents, options):
                _ = (run_id, documents, options)
                return _TestResult(success=True)
```
