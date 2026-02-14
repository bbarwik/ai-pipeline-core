#!/usr/bin/env python3
"""Showcase of ai_pipeline_core.prompt_compiler features.

Demonstrates every capability of the Prompt Compiler module:
  - Role: LLM persona definition
  - Rule: Behavioral constraints (max 5 lines)
  - OutputRule: Output formatting constraints (max 5 lines)
  - Guide: File-backed reference material (validated at import time)
  - PromptSpec: Typed prompt specification with import-time validation
  - Phase: Pipeline phase enum (planning/writing/review)
  - render_text(): Render spec instance to prompt string
  - render_preview(): Render spec class with placeholder values
  - send_spec(): Single-call LLM interaction (requires LLM proxy)
  - Shared tuples: Reuse config across specs without inheritance
  - Structured output: BaseModel as output_type for generate_structured()
  - Multi-turn: render_text() with include_input_documents=False for follow-ups
  - Import-time validation: Catches errors before runtime

No LLM connection required — all examples use render_text() and render_preview().

Usage:
  python examples/showcase_prompt_compiler.py
"""

from pydantic import BaseModel, Field

from ai_pipeline_core.documents import Document
from ai_pipeline_core.prompt_compiler import (
    Guide,
    OutputRule,
    Phase,
    PromptSpec,
    Role,
    Rule,
    extract_result,
    render_preview,
    render_text,
)

# =============================================================================
# 1. Document Types — what gets passed as context
# =============================================================================


class InitialWhitepaperDocument(Document):
    """The initial whitepaper provided by the project team."""


class PreliminaryResearchDocument(Document):
    """Preliminary research gathered from public sources."""


class IntermediateResearchDocument(Document):
    """Targeted research to investigate specific issues identified earlier."""


# =============================================================================
# 2. Roles — the identity the LLM assumes
# =============================================================================


class SeniorVCAnalyst(Role):
    """Debate participant in step 09 issue/opportunity analysis."""

    text = "Senior analyst at a top-tier venture capital firm"


class ResearchSupervisor(Role):
    """Final synthesis role that produces authoritative reports."""

    text = "Senior research supervisor responsible for final assessment quality"


# =============================================================================
# 3. Rules — behavioral constraints (max 5 lines each)
# =============================================================================


class BeAnalytical(Rule):
    """Ensures constructive tone in debate specs."""

    text = "Be analytical and constructive, not dismissive"


class FocusOnThisIssueOnly(Rule):
    """Prevents scope creep in single-issue analysis."""

    text = "Focus ONLY on the current issue — ignore other issues from research"


class CiteEvidence(Rule):
    """Requires grounding claims in document evidence."""

    text = """\
        Every claim must reference specific evidence from the provided documents.
        Use document IDs [ABC123] when citing."""


class NoSpeculation(Rule):
    """Prevents unfounded predictions."""

    text = "Do not speculate beyond what the evidence supports"


# =============================================================================
# 4. Output Rules — output formatting constraints
# =============================================================================


class StartWithOptimisticAnalyst(OutputRule):
    """Labels response for debate transcript parsing (optimistic side)."""

    text = 'Start response with "**Optimistic Analyst:**"'


class StartWithPessimisticAnalyst(OutputRule):
    """Labels response for debate transcript parsing (pessimistic side)."""

    text = 'Start response with "**Pessimistic Analyst:**"'


class DontUseMarkdownTables(OutputRule):
    """Prevents markdown tables which LLMs often format poorly."""

    text = "Do not use markdown tables — use bullet lists instead"


# =============================================================================
# 5. Guides — file-backed reference material (validated at import time)
# =============================================================================


class RiskAssessmentFramework(Guide):
    """Risk dimensions: Time Horizon, Likelihood, Impact, Complexity.
    Used by all issue debate specs."""

    template = "guides/risk_assessment_framework.txt"


class PositiveTeamAssumptions(Guide):
    """Baseline stakeholder assumptions for fair assessment.
    Used across both issue and opportunity analysis."""

    template = "guides/positive_team_assumptions.txt"


# =============================================================================
# 6. Shared Configuration — reuse without inheritance
# =============================================================================

STEP09_DOCUMENTS = (InitialWhitepaperDocument, PreliminaryResearchDocument, IntermediateResearchDocument)
STEP09_ISSUE_GUIDES = (RiskAssessmentFramework, PositiveTeamAssumptions)
STEP09_COMMON_RULES = (BeAnalytical, FocusOnThisIssueOnly, CiteEvidence)


# =============================================================================
# 7. PromptSpec — Full-featured text output spec
# =============================================================================


class IssueOptimisticSpec(PromptSpec, phase=Phase("review")):
    """Argue the optimistic case for a risk issue in the intermediate review debate."""

    input_documents = STEP09_DOCUMENTS
    role = SeniorVCAnalyst
    task = """\
        Present the OPTIMISTIC perspective on this issue.
        Argue for Medium/Long time horizon when evidence supports it.
        Show that complexity is lower than initially assessed.
        Identify concrete mitigation paths from research evidence.
        """
    guides = STEP09_ISSUE_GUIDES
    rules = STEP09_COMMON_RULES
    output_rules = (StartWithOptimisticAnalyst,)

    item: str = Field(description="The issue to analyze")


class IssuePessimisticSpec(PromptSpec, phase=Phase("review")):
    """Argue the pessimistic case for a risk issue in the intermediate review debate."""

    input_documents = STEP09_DOCUMENTS
    role = SeniorVCAnalyst
    task = """\
        Present the PESSIMISTIC perspective on this issue.
        Argue for Short time horizon when evidence supports it.
        Show that impact is higher than initially assessed.
        Identify why standard mitigations may be insufficient.
        """
    guides = STEP09_ISSUE_GUIDES
    rules = STEP09_COMMON_RULES
    output_rules = (StartWithPessimisticAnalyst,)

    item: str = Field(description="The issue to analyze")


# =============================================================================
# 8. PromptSpec — Text output with output_structure
# =============================================================================


class IssueSummarySpec(PromptSpec, phase=Phase("review")):
    """Synthesize optimistic and pessimistic arguments into a final issue assessment."""

    input_documents = STEP09_DOCUMENTS
    role = ResearchSupervisor
    task = """\
        Synthesize the debate arguments into a balanced final assessment.
        Weigh evidence from both sides fairly.
        Provide a clear recommendation with confidence level.
        """
    guides = (RiskAssessmentFramework,)
    rules = (CiteEvidence, NoSpeculation)
    output_rules = (DontUseMarkdownTables,)
    output_structure = """\
        ## 1. Executive Summary
        ## 2. Issue Profile
        ### 2.1 Time Horizon Assessment
        ### 2.2 Likelihood Assessment
        ### 2.3 Impact Assessment
        ### 2.4 Mitigation Complexity
        ## 3. Recommendations
        """

    item: str = Field(description="The issue to analyze")
    project_name: str = Field(description="Project name for context")


# =============================================================================
# 9. PromptSpec — Structured output (BaseModel)
# =============================================================================


class RiskVerdict(BaseModel):
    """Structured risk assessment result."""

    time_horizon: str
    likelihood: str
    impact_severity: str
    mitigation_complexity: str
    overall_risk_level: str
    recommendation: str


class IssueVerdictSpec(PromptSpec[RiskVerdict], phase=Phase("review")):
    """Produce a structured risk verdict for a single issue."""

    input_documents = STEP09_DOCUMENTS
    role = ResearchSupervisor
    task = """\
        Based on the debate synthesis, produce a structured risk verdict.
        Assess each dimension of the risk assessment framework.
        """
    guides = (RiskAssessmentFramework,)
    rules = (NoSpeculation,)

    item: str = Field(description="The issue to assess")


# =============================================================================
# 10. PromptSpec — Minimal spec (no optional fields, no dynamic inputs)
# =============================================================================


class WarmupAcknowledgementSpec(PromptSpec, phase=Phase("review")):
    """Warmup prompt to populate cache before forking into parallel debate calls."""

    input_documents = STEP09_DOCUMENTS
    role = SeniorVCAnalyst
    task = "Acknowledge that you have read and understood the provided documents."


# =============================================================================
# 11. PromptSpec — Different phase (WRITING)
# =============================================================================


class DraftReportSpec(PromptSpec, phase=Phase("writing")):
    """Draft a research report section based on analysis findings."""

    input_documents = (IntermediateResearchDocument,)
    role = ResearchSupervisor
    task = """\
        Draft the specified section of the research report.
        Use findings from the intermediate research as the primary source.
        """
    rules = (CiteEvidence,)

    section_title: str = Field(description="Report section to draft")
    key_findings: str = Field(description="Bullet points of key findings to incorporate")


# =============================================================================
# 12. PromptSpec — XML-wrapped response (extract_result)
# =============================================================================


class FinalVerdictSpec(PromptSpec, phase=Phase("review")):
    """Produce a final go/no-go verdict with XML-wrapped response for extraction."""

    input_documents = STEP09_DOCUMENTS
    role = ResearchSupervisor
    task = """\
        Based on all available evidence, produce a final verdict
        on whether this project should proceed to the next stage.
        """
    rules = (CiteEvidence, NoSpeculation)
    output_rules = (DontUseMarkdownTables,)
    xml_wrapped = True
    output_structure = """\
        ## Verdict
        ## Key Evidence
        ## Risk Summary
        ## Recommendation
        """

    project_name: str = Field(description="Project name")


# =============================================================================
# Demo execution
# =============================================================================


def main() -> None:
    print("=" * 80)
    print("PROMPT COMPILER SHOWCASE")
    print("=" * 80)

    # --- Feature: render_text with dynamic values ---
    print("\n--- 1. render_text(): Full prompt with dynamic values ---\n")
    spec = IssueOptimisticSpec(item="Token liquidity: Low trading volume across major DEXs")
    text = render_text(spec)
    print(text)

    # --- Feature: render_preview with placeholders ---
    print("\n" + "=" * 80)
    print("\n--- 2. render_preview(): Prompt template with placeholders ---\n")
    preview = render_preview(IssueSummarySpec)
    print(preview)

    # --- Feature: render_text without document listing (multi-turn follow-up) ---
    print("\n" + "=" * 80)
    print("\n--- 3. render_text(include_input_documents=False): Follow-up turn ---\n")
    followup = IssuePessimisticSpec(item="Token liquidity: Low trading volume across major DEXs")
    text_no_docs = render_text(followup, include_input_documents=False)
    print(text_no_docs)

    # --- Feature: Structured output spec ---
    print("\n" + "=" * 80)
    print("\n--- 4. Structured output spec (output_type=BaseModel) ---\n")
    verdict_spec = IssueVerdictSpec(item="Smart contract upgrade authority unclear")
    verdict_text = render_text(verdict_spec)
    print(verdict_text)
    print(f"\n[output_type = {IssueVerdictSpec.output_type.__name__}]")
    print("[send_spec() would use send_structured() automatically]")

    # --- Feature: Minimal spec with no dynamic fields ---
    print("\n" + "=" * 80)
    print("\n--- 5. Minimal spec (no dynamic fields, no guides, no rules) ---\n")
    warmup_text = render_text(WarmupAcknowledgementSpec())
    print(warmup_text)

    # --- Feature: Multiple dynamic fields + output_structure ---
    print("\n" + "=" * 80)
    print("\n--- 6. Multiple dynamic fields + output_structure ---\n")
    summary = IssueSummarySpec(
        item="Token liquidity: Low trading volume across major DEXs",
        project_name="DeFi Bridge Protocol",
    )
    summary_text = render_text(summary)
    print(summary_text)

    # --- Feature: Different phase ---
    print("\n" + "=" * 80)
    print("\n--- 7. WRITING phase spec ---\n")
    draft = DraftReportSpec(
        section_title="Market Analysis",
        key_findings="- Trading volume declined 40% MoM\n- Liquidity concentrated in 2 pools",
    )
    print(render_text(draft))

    # --- Feature: render_preview without documents ---
    print("\n" + "=" * 80)
    print("\n--- 8. render_preview(include_input_documents=False) ---\n")
    print(render_preview(IssueOptimisticSpec, include_input_documents=False))

    # --- Feature: Shared configuration reuse ---
    print("\n" + "=" * 80)
    print("\n--- 9. Shared configuration across specs ---\n")
    print(f"IssueOptimisticSpec.guides  = {tuple(g.__name__ for g in IssueOptimisticSpec.guides)}")
    print(f"IssuePessimisticSpec.guides = {tuple(g.__name__ for g in IssuePessimisticSpec.guides)}")
    print(f"IssueSummarySpec.guides     = {tuple(g.__name__ for g in IssueSummarySpec.guides)}")
    print("\nAll issue specs share STEP09_DOCUMENTS and STEP09_ISSUE_GUIDES via module-level tuples.")

    # --- Feature: Phase introspection ---
    print("\n" + "=" * 80)
    print("\n--- 10. Phase introspection ---\n")
    for spec_cls in [IssueOptimisticSpec, IssueSummarySpec, DraftReportSpec, WarmupAcknowledgementSpec]:
        print(f"  {spec_cls.__name__:30s} phase={spec_cls.phase}")

    # --- Feature: Import-time validation ---
    print("\n" + "=" * 80)
    print("\n--- 11. Import-time validation examples ---\n")

    errors: list[tuple[str, str]] = []

    # Missing docstring
    try:

        class _NoDocSpec(PromptSpec, phase=Phase("review")):
            input_documents = ()
            role = SeniorVCAnalyst
            task = "do it"

    except TypeError as e:
        errors.append(("Missing docstring", str(e)))

    # Missing phase
    try:

        class _NoPhaseSpec(PromptSpec):
            """Test."""

            input_documents = ()
            role = SeniorVCAnalyst
            task = "do it"

    except TypeError as e:
        errors.append(("Missing phase", str(e)))

    # Bare field without Field(description=...)
    try:

        class _BareFieldSpec(PromptSpec, phase=Phase("review")):
            """Test."""

            input_documents = ()
            role = SeniorVCAnalyst
            task = "do it"
            item: str

    except TypeError as e:
        errors.append(("Bare field (no description)", str(e)))

    # OutputRule in rules (wrong tuple)
    try:

        class _WrongRuleSpec(PromptSpec, phase=Phase("review")):
            """Test."""

            input_documents = ()
            role = SeniorVCAnalyst
            task = "do it"
            rules = (StartWithOptimisticAnalyst,)  # OutputRule in rules!

    except TypeError as e:
        errors.append(("OutputRule in rules", str(e)))

    # output_structure with BaseModel output_type
    try:

        class _BadStructSpec(PromptSpec[RiskVerdict], phase=Phase("review")):
            """Test."""

            input_documents = ()
            role = SeniorVCAnalyst
            task = "do it"
            output_structure = "## Section"

    except TypeError as e:
        errors.append(("output_structure with BaseModel", str(e)))

    # Empty task
    try:

        class _EmptyTaskSpec(PromptSpec, phase=Phase("review")):
            """Test."""

            input_documents = ()
            role = SeniorVCAnalyst
            task = "   "

    except TypeError as e:
        errors.append(("Empty task", str(e)))

    for label, msg in errors:
        print(f"  [{label}]")
        print(f"    {msg}\n")

    # --- Feature: Guide.render() reads file content ---
    print("=" * 80)
    print("\n--- 12. Guide.render() — file content ---\n")
    print(f"RiskAssessmentFramework template: {RiskAssessmentFramework.template}")
    content = RiskAssessmentFramework.render()
    print(f"Content ({len(content)} chars, first 3 lines):")
    for line in content.strip().splitlines()[:3]:
        print(f"  {line}")

    # --- Feature: send_spec usage (shown, not executed — requires LLM) ---
    print("\n" + "=" * 80)
    print("\n--- 13. send_spec() usage pattern (not executed) ---\n")
    print("""\
    # Single-call LLM interaction:
    conv = await send_spec(
        IssueOptimisticSpec(item="Token liquidity: Low volume"),
        model="gemini-3-flash",
        documents=[whitepaper, research1, research2],
    )
    print(conv.content)  # text response

    # Structured output:
    conv = await send_spec(
        IssueVerdictSpec(item="Smart contract authority"),
        model="gemini-3-pro",
        documents=docs,
    )
    print(conv.parsed)  # RiskVerdict instance

    # Multi-turn with follow-up spec:
    conv = await send_spec(warmup_spec, model=model, documents=docs)
    conv = await conv.send(render_text(optimistic_spec, include_input_documents=False))

    # Warmup + fork for parallel calls:
    warmup = await send_spec(warmup_spec, model=model, documents=docs)
    fork1, fork2 = await asyncio.gather(
        warmup.send(render_text(optimistic_spec, include_input_documents=False)),
        warmup.send(render_text(pessimistic_spec, include_input_documents=False)),
    )""")

    # --- Feature: XML-wrapped response spec ---
    print("\n" + "=" * 80)
    print("\n--- 14. XML-wrapped response spec (xml_wrapped=True) ---\n")
    verdict = FinalVerdictSpec(project_name="DeFi Bridge Protocol")
    verdict_text = render_text(verdict)
    print(verdict_text)

    # --- Feature: extract_result utility ---
    print("\n" + "=" * 80)
    print("\n--- 15. extract_result() utility ---\n")
    sample_response = "Some preamble\n<result>\n## Verdict\nProceed with caution.\n</result>\nTrailing text"
    extracted = extract_result(sample_response)
    print(f"Input:     {sample_response!r}")
    print(f"Extracted: {extracted!r}")
    no_tags = "Plain response without tags"
    print(f"\nNo tags:   {extract_result(no_tags)!r} (returns as-is)")

    # --- Feature: xml_wrapped validation ---
    print("\n" + "=" * 80)
    print("\n--- 16. xml_wrapped import-time validation ---\n")

    xml_errors: list[tuple[str, str]] = []

    # xml_wrapped with BaseModel output_type
    try:

        class _XmlBaseModelSpec(PromptSpec[RiskVerdict], phase=Phase("review")):
            """Test."""

            input_documents = ()
            role = SeniorVCAnalyst
            task = "do it"
            xml_wrapped = True

    except TypeError as e:
        xml_errors.append(("xml_wrapped with BaseModel", str(e)))

    # H1 header in output_structure
    try:

        class _H1OutputSpec(PromptSpec, phase=Phase("review")):
            """Test."""

            input_documents = ()
            role = SeniorVCAnalyst
            task = "do it"
            output_structure = "# Bad Header"

    except TypeError as e:
        xml_errors.append(("H1 in output_structure", str(e)))

    for label, msg in xml_errors:
        print(f"  [{label}]")
        print(f"    {msg}\n")

    print("=" * 80)
    print("\nAll features demonstrated successfully.")


if __name__ == "__main__":
    main()
