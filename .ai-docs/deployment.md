# MODULE: deployment
# CLASSES: DeploymentResult, FlowAction, FlowDirective, PipelineDeployment, RemoteDeployment
# DEPENDS: BaseModel, Generic, StrEnum
# PURPOSE: Pipeline deployment utilities for unified, type-safe deployments.
# VERSION: 0.19.2
# AUTO-GENERATED from source code — do not edit. Run: make docs-ai-build

## Imports

```python
from ai_pipeline_core import DeploymentResult, PipelineDeployment, RemoteDeployment
from ai_pipeline_core.deployment import FlowAction, FlowDirective
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
    flow_retries: ClassVar[int | None] = None
    flow_retry_delay_seconds: ClassVar[int | None] = None
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
        run_id: str | None,
        documents: Sequence[Document],
        options: TOptions,
        publisher: ResultPublisher | None = None,
        output_dir: Path | None = None,
    ) -> TResult:
        """Run locally with Prefect test harness and in-memory database.

        Args:
            run_id: Pipeline run identifier. If None, auto-generated from output_dir + date + input hash.
            documents: Initial input documents.
            options: Flow options.
            publisher: Optional lifecycle event publisher (defaults to _NoopPublisher).
            output_dir: Optional directory for writing result.json.

        Returns:
            Typed deployment result.
        """
        if run_id is None:
            dir_name = output_dir.name if output_dir else "local"
            run_id = build_auto_run_id(output_dir_name=dir_name, documents=documents, options=options)

        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)

        with prefect_test_harness(), disable_run_logger():
            result = asyncio.run(self.run(run_id, documents, options, publisher=publisher, database=_MemoryDatabase()))

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
        if self.deployment_class:
            use_inline = True
            inline_reason = "deployment_class is set (inline execution configured)"

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


```

## Examples

**Format starts with base run id** (`tests/deployment/test_remote_deployment.py:851`)

```python
class AlphaDoc(Document):
    """First test document type."""


def test_format_starts_with_base_run_id(self):
    """Derived run_id starts with the user's base run_id."""
    doc = AlphaDoc.create_root(name="test.txt", content="hello", reason="test")
    derived = _derive_remote_run_id("my-project", [doc], FlowOptions())
    assert derived.startswith("my-project-")
```

**Deployment default flow retries is none** (`tests/deployment/test_flow_retries.py:230`)

```python
def test_deployment_default_flow_retries_is_none(self) -> None:
    assert PipelineDeployment.flow_retries is None
    assert PipelineDeployment.flow_retry_delay_seconds is None
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

**Extracts remote deployment params** (`tests/deployment/test_helpers.py:57`)

```python
def test_extracts_remote_deployment_params(self):
    """Test correct extraction from RemoteDeployment subclass (2 params)."""
    params = extract_generic_params(SampleRemote, RemoteDeployment)
    assert len(params) == 2
    assert params[0] is FlowOptions
    assert params[1] is SampleResult
```

**Two args returned by helper** (`tests/deployment/test_remote_deployment.py:126`)

```python
def test_two_args_returned_by_helper(self):
    class Foo(RemoteDeployment[FlowOptions, SimpleResult]):
        pass

    args = extract_generic_params(Foo, RemoteDeployment)
    assert len(args) == 2
    assert args[0] is FlowOptions
    assert args[1] is SimpleResult
```

**Two params from remote deployment** (`tests/deployment/test_remote_deployment.py:798`)

```python
def test_two_params_from_remote_deployment(self):
    class Foo(RemoteDeployment[FlowOptions, SimpleResult]):
        pass

    result = extract_generic_params(Foo, RemoteDeployment)
    assert len(result) == 2
    assert result[0] is FlowOptions
    assert result[1] is SimpleResult
```

**Accepts deployment result subclass** (`tests/deployment/test_remote_deployment.py:166`)

```python
def test_accepts_deployment_result_subclass(self):
    class Foo(RemoteDeployment[FlowOptions, SimpleResult]):
        pass

    assert Foo.result_type is SimpleResult
```

**Accepts flow options subclass** (`tests/deployment/test_remote_deployment.py:146`)

```python
def test_accepts_flow_options_subclass(self):
    class CustomOpts(FlowOptions):
        budget: float = 10.0

    class Good(RemoteDeployment[CustomOpts, SimpleResult]):
        pass

    assert Good.options_type is CustomOpts
```


## Error Examples

**Rejects empty run id** (`tests/deployment/test_remote_deployment.py:515`)

```python
async def test_rejects_empty_run_id(self):
    class Foo(RemoteDeployment[FlowOptions, SimpleResult]):
        pass

    with pytest.raises(ValueError, match="must not be empty"):
        with pipeline_test_context(run_id=""):
            await Foo().run((), FlowOptions())
```

**Rejects invalid run id** (`tests/deployment/test_remote_deployment.py:507`)

```python
async def test_rejects_invalid_run_id(self):
    class Foo(RemoteDeployment[FlowOptions, SimpleResult]):
        pass

    with pytest.raises(ValueError, match="contains invalid characters"):
        with pipeline_test_context(run_id="invalid run id with spaces"):
            await Foo().run((), FlowOptions())
```

**Rejects no generic params** (`tests/deployment/test_remote_deployment.py:174`)

```python
def test_rejects_no_generic_params(self):
    with pytest.raises(TypeError, match="must specify 2 Generic parameters"):

        class Bad(RemoteDeployment):  # type: ignore[type-arg]
            pass
```

**Rejects non deployment result** (`tests/deployment/test_remote_deployment.py:157`)

```python
def test_rejects_non_deployment_result(self):
    class NotAResult(BaseModel):
        x: int = 1

    with pytest.raises(TypeError, match="DeploymentResult subclass"):

        class Bad(RemoteDeployment[FlowOptions, NotAResult]):  # type: ignore[type-var]
            pass
```

**Rejects non flow options** (`tests/deployment/test_remote_deployment.py:137`)

```python
def test_rejects_non_flow_options(self):
    class NotFlowOptions(BaseModel):
        x: int = 1

    with pytest.raises(TypeError, match="FlowOptions subclass"):

        class Bad(RemoteDeployment[NotFlowOptions, SimpleResult]):  # type: ignore[type-var]
            pass
```

**Rejects outside execution context** (`tests/deployment/test_remote_deployment.py:500`)

```python
async def test_rejects_outside_execution_context(self):
    class Foo(RemoteDeployment[FlowOptions, SimpleResult]):
        pass

    with pytest.raises(RuntimeError, match="pipeline_test_context"):
        await Foo().run((), FlowOptions())
```
