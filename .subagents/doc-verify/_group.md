---
description: >
  Verify quality of auto-generated .ai-docs/ by running 5 specialized agents against selected module guides.
  README.md is always included as shared context. Pass additional module guides via context parameter,
  e.g. context=[".ai-docs/llm.md", ".ai-docs/documents.md"]. At least one module guide must be provided.
models: ["gemini-3-flash-preview", "gpt-5.3-codex-spark", "haiku"]
read_only: true
timeout_minutes: 5
context: ["README.md"]
---
You are reviewing auto-generated API documentation for a Python framework called ai-pipeline-core.
You MUST read the documentation files provided in your context using the file reading tool — these are the files you need to analyze.
Do NOT use any other tools besides reading the provided files. Do NOT run commands, write files, or search the web.
Do NOT read any files other than the ones explicitly provided in your context.
Be specific and thorough. Quote exact text from the docs when referencing them. Explain your reasoning clearly.
