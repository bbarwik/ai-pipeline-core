"""Class-based pipeline flow runtime and validation."""

import annotationlib
import ast
import inspect
import math
import textwrap
from collections.abc import Callable
from typing import Any, ClassVar, cast, get_origin

from ai_pipeline_core.documents import Document
from ai_pipeline_core.logger import get_pipeline_logger
from ai_pipeline_core.pipeline._task import PipelineTask
from ai_pipeline_core.pipeline._type_validation import collect_document_types, contains_bare_document, resolve_type_hints
from ai_pipeline_core.pipeline.options import FlowOptions

logger = get_pipeline_logger(__name__)

__all__ = [
    "PipelineFlow",
]


def _declared_init_annotations(klass: type) -> set[str]:
    """Return non-private annotations declared directly on ``klass``."""
    annotate = annotationlib.get_annotate_from_class_namespace(klass.__dict__)
    if callable(annotate):
        annotations = cast(dict[str, Any], annotationlib.call_annotate_function(cast(Any, annotate), format=annotationlib.Format.FORWARDREF))
        return {name for name in annotations if not name.startswith("_")}

    annotations = annotationlib.get_annotations(klass, format=annotationlib.Format.FORWARDREF)
    return {name for name in annotations if not name.startswith("_")}


def _resolve_task_name(globals_dict: dict[str, Any], name: str) -> str | None:
    """Check if name refers to a PipelineTask subclass in the given globals."""
    candidate = globals_dict.get(name)
    if isinstance(candidate, type) and issubclass(candidate, PipelineTask) and candidate is not PipelineTask:
        return candidate.name
    return None


def _extract_task_name_from_call(node: ast.Call, globals_dict: dict[str, Any]) -> str | None:
    """Extract task class name from supported task call patterns."""
    if isinstance(node.func, ast.Attribute) and node.func.attr == "run" and isinstance(node.func.value, ast.Name):
        return _resolve_task_name(globals_dict, node.func.value.id)
    return None


def _parse_task_graph_from_source(run_fn: Any) -> list[tuple[str, str]]:
    """Extract task invocations from flow run() source.

    Recognizes:
    - `await TaskClass.run(...)` as sequential
    - `TaskClass.run(...)` as deferred / handle-producing
    """
    try:
        source = textwrap.dedent(inspect.getsource(run_fn))
        tree = ast.parse(source)
    except OSError, TypeError, SyntaxError:
        return []

    graph: list[tuple[str, str]] = []
    globals_dict = getattr(run_fn, "__globals__", {})

    awaited_calls: set[int] = set()

    class _Visitor(ast.NodeVisitor):
        def visit_Await(self, node: ast.Await) -> None:
            if isinstance(node.value, ast.Call):
                task_name = _extract_task_name_from_call(node.value, globals_dict)
                if task_name is not None:
                    graph.append((task_name, "sequential"))
                    awaited_calls.add(id(node.value))
            self.generic_visit(node)

        def visit_Call(self, node: ast.Call) -> None:
            if id(node) not in awaited_calls:
                task_name = _extract_task_name_from_call(node, globals_dict)
                if task_name is not None:
                    graph.append((task_name, "dispatched"))
            self.generic_visit(node)

    _Visitor().visit(tree)
    return graph


class PipelineFlow:
    """Base class for pipeline flows.

    Flows are the unit of resume, progress tracking, and document hand-off in a deployment.
    Define ``run`` as an **instance method** (not @classmethod) because flows can carry
    per-instance configuration passed via ``build_flows()``::

        class TranslateFlow(PipelineFlow):
            target_language: str = "en"

            async def run(self, documents: tuple[SourceDoc, ...], options: FlowOptions) -> tuple[TranslatedDoc, ...]:
                return await TranslateTask.run(documents, language=self.target_language)

    The deployment creates flow instances with constructor kwargs::

        def build_flows(self, options):
            return [TranslateFlow(target_language="fr"), TranslateFlow(target_language="de")]

    Each instance runs independently with its own parameters, resume record, and progress.
    Constructor kwargs are captured for replay serialization via ``get_params()``.

    Signature must be exactly ``(self, documents: tuple[DocType, ...], options: FlowOptions)``
    and is validated at class definition time by ``__init_subclass__``.
    Use ``get_run_id()`` from ``ai_pipeline_core.pipeline`` to access the run ID inside a flow.
    """

    name: ClassVar[str]
    estimated_minutes: ClassVar[float] = 1.0
    BASE_COST_USD: ClassVar[float] = 0.0
    input_document_types: ClassVar[list[type[Document]]] = []
    output_document_types: ClassVar[list[type[Document]]] = []
    task_graph: ClassVar[list[tuple[str, str]]] = []

    async def run(self, documents: tuple[Any, ...], options: Any) -> tuple[Any, ...]:
        """Execute the flow.

        Subclasses must provide concrete ``tuple[MyDocument, ...]`` and ``FlowOptions``
        annotations. The base stub stays broad so static type checkers accept
        narrower overrides; ``__init_subclass__`` enforces the real signature.
        """
        raise NotImplementedError

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if cls is PipelineFlow:
            return

        cls._validate_class_config()
        run_fn, hints, params = cls._validate_run_signature()
        input_types, output_types = cls._extract_document_types(hints, params)
        cls.input_document_types = input_types
        cls.output_document_types = output_types
        cls.task_graph = cls._parse_task_graph(run_fn)

    @classmethod
    def _validate_class_config(cls) -> None:
        if cls.__name__.startswith("Test"):
            raise TypeError(
                f"PipelineFlow class name cannot start with 'Test': {cls.__name__}. Use a production-style class name; pytest classes reserve the Test* prefix."
            )
        if "name" not in cls.__dict__:
            cls.name = cls.__name__
        if cls.estimated_minutes < 1:
            raise TypeError(f"PipelineFlow '{cls.__name__}' has estimated_minutes={cls.estimated_minutes}. Use a value >= 1.")
        if not math.isfinite(cls.BASE_COST_USD) or cls.BASE_COST_USD < 0:
            raise TypeError(
                f"PipelineFlow '{cls.__name__}' has BASE_COST_USD={cls.BASE_COST_USD}. "
                "Use a finite float >= 0 representing the minimum cost in USD per execution."
            )

    @classmethod
    def _validate_run_signature(cls) -> tuple[Callable[..., Any], dict[str, Any], list[inspect.Parameter]]:
        run_fn = cls.__dict__.get("run")
        if run_fn is None:
            # Check for inherited run from a parent subclass (not the base PipelineFlow stub)
            for parent in cls.__mro__[1:]:
                if parent is PipelineFlow:
                    break
                if "run" in parent.__dict__:
                    run_fn = parent.__dict__["run"]
                    break
        if run_fn is None:
            raise TypeError(f"PipelineFlow '{cls.__name__}' must define async run(self, documents, options) -> tuple[Document, ...].")
        if not inspect.iscoroutinefunction(run_fn):
            raise TypeError(f"PipelineFlow '{cls.__name__}'.run must be async def. Use async operations in flow code and return tuple[Document, ...].")

        sig = inspect.signature(run_fn)
        params = list(sig.parameters.values())
        if len(params) != 3:
            raise TypeError(
                f"PipelineFlow '{cls.__name__}'.run must have signature "
                f"(self, documents: tuple[DocType, ...], options: FlowOptions) -> tuple[DocType, ...]. "
                f"Found parameters: {', '.join(p.name for p in params)}. "
                f"run_id is no longer a parameter — use get_run_id() from ai_pipeline_core.pipeline."
            )
        hints = resolve_type_hints(run_fn)
        return run_fn, hints, params

    @classmethod
    def _extract_document_types(
        cls,
        hints: dict[str, Any],
        params: list[inspect.Parameter],
    ) -> tuple[list[type[Document]], list[type[Document]]]:
        documents_param = params[1].name
        documents_annotation = hints.get(documents_param)
        if documents_annotation is None:
            raise TypeError(
                f"PipelineFlow '{cls.__name__}'.run is missing type annotation for '{documents_param}'. Use tuple[MyDocument, ...] or tuple[DocA | DocB, ...]."
            )
        if contains_bare_document(documents_annotation):
            raise TypeError(f"PipelineFlow '{cls.__name__}' uses bare 'Document' in run() input. Use specific Document subclasses in tuple[...] annotations.")
        if get_origin(documents_annotation) is not tuple:
            raise TypeError(
                f"PipelineFlow '{cls.__name__}'.run input must be annotated as "
                f"tuple[DocumentSubclass, ...] or tuple[DocA | DocB, ...]. "
                f"Got: {documents_annotation!r}."
            )
        input_types = collect_document_types(documents_annotation)
        if not input_types:
            raise TypeError(
                f"PipelineFlow '{cls.__name__}'.run input must be annotated as "
                f"tuple[DocumentSubclass, ...] or tuple[DocA | DocB, ...]. "
                f"Got: {documents_annotation!r}."
            )

        options_param = params[2].name
        options_type = hints.get(options_param)
        if not (isinstance(options_type, type) and issubclass(options_type, FlowOptions)):
            raise TypeError(f"PipelineFlow '{cls.__name__}'.run parameter '{options_param}' must be FlowOptions or subclass. Got: {options_type!r}.")

        return_annotation = hints.get("return")
        if return_annotation is None:
            raise TypeError(f"PipelineFlow '{cls.__name__}'.run is missing return annotation. Use tuple[MyDocument, ...] or tuple[DocA | DocB, ...].")
        if contains_bare_document(return_annotation):
            raise TypeError(
                f"PipelineFlow '{cls.__name__}' uses bare 'Document' in run() return type. Use specific Document subclasses in tuple[...] annotations."
            )
        if get_origin(return_annotation) is not tuple:
            raise TypeError(
                f"PipelineFlow '{cls.__name__}'.run must return tuple[DocumentSubclass, ...] or tuple[DocA | DocB, ...]. Got: {return_annotation!r}."
            )
        output_types = collect_document_types(return_annotation)
        if not output_types:
            raise TypeError(
                f"PipelineFlow '{cls.__name__}'.run must return tuple[DocumentSubclass, ...] or tuple[DocA | DocB, ...]. Got: {return_annotation!r}."
            )
        overlap = set(input_types) & set(output_types)
        if overlap:
            raise TypeError(
                f"PipelineFlow '{cls.__name__}' has overlapping input/output document types "
                f"({', '.join(sorted(t.__name__ for t in overlap))}). "
                "A flow must not both consume and produce the same document type."
            )
        return input_types, output_types

    @classmethod
    def _parse_task_graph(cls, run_fn: Callable[..., Any]) -> list[tuple[str, str]]:
        return _parse_task_graph_from_source(run_fn)

    def __init__(self, **kwargs: Any) -> None:
        """Constructor for per-flow instance configuration."""
        cls = type(self)
        known_params: set[str] = set()
        for klass in cls.__mro__:
            known_params.update(_declared_init_annotations(klass))
            known_params.update(
                name
                for name, value in vars(klass).items()
                if not name.startswith("_") and not callable(value) and not isinstance(value, (classmethod, staticmethod, property))
            )
        unknown = sorted(key for key in kwargs if key not in known_params)
        if unknown:
            allowed = ", ".join(sorted(known_params)) or "(none)"
            raise TypeError(f"PipelineFlow '{cls.__name__}' got unknown init parameter(s): {', '.join(unknown)}. Allowed parameters: {allowed}.")
        self._params: dict[str, Any] = dict(kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)

    def get_params(self) -> dict[str, Any]:
        """Return constructor params for flow plan serialization."""
        return dict(getattr(self, "_params", {}))

    @classmethod
    def expected_tasks(cls) -> list[str]:
        """Return expected task names extracted from run() AST."""
        return [name for name, _mode in cls.task_graph]
