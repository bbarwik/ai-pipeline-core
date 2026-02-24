# Prompt Validation Rules

> Rules for validating PromptSpec definitions. Intended as instructions for an LLM-based validator
> that reviews compiled prompts via the `ai-prompt-compiler validate` CLI command.

The validator receives a rendered prompt (output of `render_text()`) together with the PromptSpec
class metadata (role, task, guides, rules, output_rules, output_type, output_structure, input_documents,
dynamic fields). It checks the prompt against the rules below and reports violations.

---

## 1. Role–Task Alignment

**What to check:** The Role's text must match the scope of the task.

- If the task handles a single domain (e.g., only risks, only opportunities), a domain-specific
  Role is appropriate (e.g., `RiskStrategist`).
- If the task is parameterized by domain (dynamic field controls whether it processes risks,
  opportunities, questions, or other categories), the Role must be domain-neutral
  (e.g., `DueDiligenceAnalyst`, not `RiskStrategist`).
- A narrow Role applied to a multi-domain spec biases LLM output toward that domain's framing.

**How to detect:** Check if the task text or dynamic field descriptions reference multiple domains
or categories. If they do, verify the Role text does not favor one domain over others.

**Violation example:** Role is "risk strategist" but task says "analyze the following {finding_type}"
where finding_type can be "risk", "opportunity", or "question".

---

## 2. Guide Completeness

**What to check:** Every domain-specific term referenced in the task text must be defined by an
attached Guide — or be a universally understood term.

- Scan the task text for title-case multi-word phrases and capitalized terms that look like
  framework-specific vocabulary (e.g., "Novelty", "Testability", "Effort Level", "Value Score").
- If the term matches a Guide class name or a heading within a Guide's rendered content, that
  Guide must be in the `guides` tuple.
- Missing Guides cause the LLM to hallucinate definitions for framework-specific terms.

**How to detect:** Extract capitalized terms from task text. Cross-reference against Guide class
names and Guide content headings. Flag terms that appear in existing Guide content but whose Guide
is not attached to this spec.

**Violation example:** Task says "assess Novelty, Testability, and Value" but only
`PositiveTeamAssumptionsGuide` is attached. `OpportunitiesAssessmentGuide` (which defines
Novelty, Testability, Value) is missing.

---

## 3. Task–Output Field Alignment

**What to check:** The vocabulary used in the task text must align with the field names and
descriptions in the structured output model (when `output_type` is a BaseModel).

- If the task uses domain-specific terms (e.g., "Novelty", "Effort", "Value") but the output
  model has generic fields (e.g., `likelihood`, `impact`, `complexity`), explicit mapping
  instructions must be present in the task.
- Without explicit mapping, the LLM guesses how domain terms map to generic fields, producing
  inconsistent results.

**How to detect:** Extract key terms from the task text. Compare against output model field names
and their `Field(description=...)` values. If there is vocabulary mismatch with no explicit
mapping instruction ("map X to field Y"), flag it.

**Violation example:** Task says "evaluate Novelty and Effort" but output model has fields
`likelihood: str` and `complexity: str` with no mapping instruction.

---

## 4. XML Wrapping Consistency

**What to check:** When `output_structure` is set, no OutputRule should instruct the LLM to add XML tags.

- `output_structure` automatically enables `<result>` wrapping and auto-extraction in `send_spec()`.
- If an OutputRule's text also references XML tags (e.g., "wrap output in `<document>` tags"),
  the LLM produces double-wrapped output like `<result><document>content</document></result>`.
- Auto-extraction strips `<result>` but leaves inner XML tags in the content.

**How to detect:** When `output_structure` is set, scan all OutputRule texts for XML tag patterns
(`<tag>`, `</tag>`, `<tag/>`). Flag any matches.

**Note:** This check is also enforced at definition time via `__init_subclass__` validation.
The validator serves as a secondary safety net.

---

## 5. Output Structure vs Output Type Conflict

**What to check:** `output_structure` (free-text format instructions) is only meaningful for
`PromptSpec[str]`. It must not be set when using structured output (`PromptSpec[MyModel]`).

- Structured output sends the BaseModel JSON schema to the LLM. Adding prose format
  instructions via `output_structure` creates conflicting guidance.
- This is already enforced at definition time, but the validator should verify the rendered
  prompt doesn't contain contradictory formatting instructions.

**How to detect:** If `output_type` is a BaseModel subclass, verify no "Output Structure" section
appears in the rendered prompt.

---

## 6. Document Declaration Consistency

**What to check:** The relationship between `input_documents` (type declarations) and actual
document usage in the task text.

- If `input_documents` declares Document types but the task text never references those
  document types (by name, description, or implied content), the declaration may be wrong.
- If the task text implies documents that aren't declared in `input_documents`, the spec is
  incomplete.
- `input_documents` are type declarations — they tell the framework what Document subclasses
  this spec expects. Actual instances are passed via `send_spec(documents=...)`.

**How to detect:** Compare Document type names and their docstrings against references in the
task text. Flag Document types declared but never referenced, and document references in the
task text that don't match any declared type.

---

## 7. Rule Contradiction Detection

**What to check:** Rules and OutputRules within the same spec must not contradict each other.

- Two rules that give opposite instructions (e.g., "include source IDs" vs "never include
  identifiers") create conflicting guidance.
- Contradictions between a spec and its follow-up are intentional —
  only flag contradictions within the same spec.

**How to detect:** Compare each pair of rules semantically. Flag pairs where one rule requires
something the other prohibits.

**Note:** This requires semantic understanding — exact detection is the LLM validator's primary
value over static analysis.

---

## 8. Decomposition Before Decision (Structured Output)

**What to check:** When `output_type` is a BaseModel, fields that decompose the problem must
be defined before fields that represent conclusions.

- LLMs generate tokens sequentially. If a decision field (`is_valid: bool`) comes before
  analysis fields (`discrepancies: str`), the LLM commits to a conclusion first and
  rationalizes afterward.
- Generic scratchpad fields (`reasoning: str`, `thinking: str`) are redundant with the model's
  native reasoning — flag them.

**How to detect:** Examine field order in the output BaseModel. Boolean/enum decision fields
should appear after string analysis fields. Flag `reasoning`, `thinking`, `analysis` as
generic scratchpad names.

---

## 9. Task Clarity and Specificity

**What to check:** The task text must give clear, actionable instructions.

- Avoid vague directives ("analyze this", "review the document") without specifying what to
  look for or what the output should contain.
- When the task references specific analytical dimensions, each dimension should be named
  explicitly rather than implied.
- Tasks should not include chain-of-thought prompting ("think step by step") — all 2026 models
  are thinking models with native reasoning.

**How to detect:** Flag task texts shorter than 2 sentences without output_structure or
output_type guidance. Flag chain-of-thought instructions. Flag tasks that reference "the
document" without specifying which document type (when multiple are declared).

---

## 10. Dynamic Field Usage

**What to check:** Every dynamic field declared on the PromptSpec must be referenced in the
rendered prompt.

- Dynamic fields (Pydantic fields with `Field(description=...)`) are rendered in the Context
  section. If a field exists but the task never uses its value, the field may be unnecessary
  or the task may be incomplete.
- Field descriptions should be specific enough for the LLM to understand the value's purpose.

**How to detect:** For each dynamic field, check if its description label or field name appears
in the task text or output_structure. Flag fields that appear only in the Context section
with no reference elsewhere in the prompt.

---

## 11. Content Embedding Anti-Pattern

**What to check:** Large content (documents, reports, web content) must be provided as Document
objects in context, not embedded in dynamic field values or task text.

- Documents in context get XML wrapping (prompt injection defense) and enable cache reuse.
- Embedding content in task text or dynamic fields creates unique prompts that cannot be cached
  and lack the XML data/instruction boundary.

**How to detect:** Flag dynamic fields whose values exceed 500 characters — they likely contain
content that should be a Document instead. Flag task text containing markdown code blocks with
embedded content.

---

## 12. Stop Sequence Awareness

**What to check:** When `output_structure` is set, verify the model supports stop sequences.

- The framework automatically adds `</result>` as a stop sequence for specs with `output_structure`.
- For unsupported models, the LLM may generate content after `</result>`, which
  auto-extraction handles but wastes output tokens.

**How to detect:** This is informational — flag as a warning (not error) when `output_structure` is set
and the target model is not in the supported stop sequence list.

---

## Validation Severity Levels

| Level | Meaning | Action |
|-------|---------|--------|
| **error** | Prompt will produce incorrect or degraded output | Must fix before using |
| **warning** | Prompt may produce suboptimal output | Should fix, but functional |
| **info** | Suggestion for improvement | Optional |

### Severity Assignment

| Rule | Default Severity |
|------|-----------------|
| 1. Role–Task Alignment | warning |
| 2. Guide Completeness | error |
| 3. Task–Output Field Alignment | error |
| 4. XML Wrapping Consistency | error |
| 5. Output Structure vs Type Conflict | error |
| 6. Document Declaration Consistency | warning |
| 7. Rule Contradiction Detection | error |
| 8. Decomposition Before Decision | warning |
| 9. Task Clarity and Specificity | info |
| 10. Dynamic Field Usage | warning |
| 11. Content Embedding Anti-Pattern | warning |
| 12. Stop Sequence Awareness | info |
