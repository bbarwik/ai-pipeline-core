"""AI-focused documentation generator.

Generates dense, self-contained guides from source code and test suite
for AI coding agents. Uses AST parsing, dependency resolution, and
size management for guides with a 50KB warning threshold.
"""

from ai_pipeline_core.docs_generator.extractor import (
    ClassInfo,
    FunctionInfo,
    MethodInfo,
    ModuleInfo,
    SymbolTable,
    is_public_name,
    parse_module,
)
from ai_pipeline_core.docs_generator.guide_builder import (
    GuideData,
    ScoredExample,
    build_guide,
    discover_tests,
    select_examples,
)
from ai_pipeline_core.docs_generator.trimmer import manage_guide_size
from ai_pipeline_core.docs_generator.validator import (
    ValidationResult,
    compute_source_hash,
    validate_all,
    validate_completeness,
    validate_freshness,
    validate_size,
)

__all__ = [
    "ClassInfo",
    "FunctionInfo",
    "GuideData",
    "MethodInfo",
    "ModuleInfo",
    "ScoredExample",
    "SymbolTable",
    "ValidationResult",
    "build_guide",
    "compute_source_hash",
    "discover_tests",
    "is_public_name",
    "manage_guide_size",
    "parse_module",
    "select_examples",
    "validate_all",
    "validate_completeness",
    "validate_freshness",
    "validate_size",
]
