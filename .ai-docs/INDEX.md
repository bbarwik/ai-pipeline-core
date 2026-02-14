# AI Documentation Index

Auto-generated guide index. Do not edit manually.

## Reading Order

1. [deployment](deployment.md) — Pipeline deployment utilities for unified, type-safe deployments.
2. [document_store](document_store.md) — Document store protocol and backends for AI pipeline flows.
3. [documents](documents.md) — Document system for AI pipeline flows.
4. [exceptions](exceptions.md)
5. [llm](llm.md) — Large Language Model integration via LiteLLM proxy.
6. [logging](logging.md) — Logging infrastructure for AI Pipeline Core.
7. [observability](observability.md) — Observability system for AI pipelines.
8. [pipeline](pipeline.md) — Pipeline framework primitives — decorators, flow options, and concurrency limits.
9. [prompt_compiler](prompt_compiler.md) — Prompt compiler for type-safe, validated prompt specifications.
10. [prompt_manager](prompt_manager.md)
11. [settings](settings.md)

## Symbol Index

| Symbol | Kind | Module |
| ------ | ---- | ------ |
| APPROX_CHARS_PER_TOKEN | constant | [prompt_compiler](prompt_compiler.md) |
| Attachment | class | [documents](documents.md) |
| camel_to_snake | func | [documents](documents.md) |
| Citation | class | [llm](llm.md) |
| ClickHouseDocumentStore | class | [document_store](document_store.md) |
| CompletedRun | class | [deployment](deployment.md) |
| Conversation | class | [llm](llm.md) |
| CoreMessage | class | [llm](llm.md) |
| create_document_store | func | [document_store](document_store.md) |
| DATA_URI_PATTERN | constant | [documents](documents.md) |
| DEFAULT_LOG_LEVELS | constant | [logging](logging.md) |
| Deployer | class | [deployment](deployment.md) |
| DeploymentContext | class | [deployment](deployment.md) |
| DeploymentResult | class | [deployment](deployment.md) |
| DeploymentResultData | class | [deployment](deployment.md) |
| detect_mime_type | func | [documents](documents.md) |
| DOC_ID_LENGTH | constant | [document_store](document_store.md) |
| Document | class | [documents](documents.md) |
| DocumentNameError | class | [exceptions](exceptions.md) |
| DocumentSha256 | newtype | [documents](documents.md) |
| DocumentSizeError | class | [exceptions](exceptions.md) |
| DocumentStore | class | [document_store](document_store.md) |
| DocumentValidationError | class | [exceptions](exceptions.md) |
| EXTENSION_MIME_MAP | constant | [documents](documents.md) |
| extract_result | func | [prompt_compiler](prompt_compiler.md) |
| FailedRun | class | [deployment](deployment.md) |
| flow_context | func | [deployment](deployment.md) |
| FlowOptions | class | [pipeline](pipeline.md) |
| FlowStatus | class | [deployment](deployment.md) |
| generate | func | [llm](llm.md) |
| generate_structured | func | [llm](llm.md) |
| get_document_store | func | [document_store](document_store.md) |
| get_pipeline_logger | func | [logging](logging.md) |
| get_tiktoken_encoding | func | [documents](documents.md) |
| Guide | class | [prompt_compiler](prompt_compiler.md) |
| ImageContent | class | [llm](llm.md) |
| is_document_sha256 | func | [documents](documents.md) |
| is_image_mime_type | func | [documents](documents.md) |
| is_json_mime_type | func | [documents](documents.md) |
| is_llm_supported_image | func | [documents](documents.md) |
| is_pdf_mime_type | func | [documents](documents.md) |
| is_text_mime_type | func | [documents](documents.md) |
| is_yaml_mime_type | func | [documents](documents.md) |
| LimitKind | class | [pipeline](pipeline.md) |
| LLMError | class | [exceptions](exceptions.md) |
| LocalDocumentStore | class | [document_store](document_store.md) |
| LoggingConfig | class | [logging](logging.md) |
| main | func | [prompt_compiler](prompt_compiler.md) |
| MAX_PROVENANCE_GRAPH_NODES | constant | [document_store](document_store.md) |
| MAX_RULE_LINES | constant | [prompt_compiler](prompt_compiler.md) |
| MemoryDocumentStore | class | [document_store](document_store.md) |
| ModelName | typealias | [llm](llm.md) |
| ModelOptions | class | [llm](llm.md) |
| ModelResponse | class | [llm](llm.md) |
| OutputRule | class | [prompt_compiler](prompt_compiler.md) |
| PDFContent | class | [llm](llm.md) |
| PendingRun | class | [deployment](deployment.md) |
| Phase | class | [prompt_compiler](prompt_compiler.md) |
| PIP_TARGET_PLATFORMS | constant | [deployment](deployment.md) |
| pipeline_concurrency | func | [pipeline](pipeline.md) |
| pipeline_flow | func | [pipeline](pipeline.md) |
| pipeline_task | func | [pipeline](pipeline.md) |
| PipelineCoreError | class | [exceptions](exceptions.md) |
| PipelineDeployment | class | [deployment](deployment.md) |
| PipelineLimit | class | [pipeline](pipeline.md) |
| ProgressContext | class | [deployment](deployment.md) |
| ProgressRun | class | [deployment](deployment.md) |
| PromptError | class | [exceptions](exceptions.md) |
| PromptManager | class | [prompt_manager](prompt_manager.md) |
| PromptNotFoundError | class | [exceptions](exceptions.md) |
| PromptRenderError | class | [exceptions](exceptions.md) |
| PromptSpec | class | [prompt_compiler](prompt_compiler.md) |
| RemoteDeployment | class | [deployment](deployment.md) |
| render_preview | func | [prompt_compiler](prompt_compiler.md) |
| render_text | func | [prompt_compiler](prompt_compiler.md) |
| RESULT_CLOSE | constant | [prompt_compiler](prompt_compiler.md) |
| RESULT_OPEN | constant | [prompt_compiler](prompt_compiler.md) |
| RESULT_TAG | constant | [prompt_compiler](prompt_compiler.md) |
| RetryConditionCallable | typealias | [pipeline](pipeline.md) |
| Role | class | [prompt_compiler](prompt_compiler.md) |
| Rule | class | [prompt_compiler](prompt_compiler.md) |
| run_remote_deployment | func | [deployment](deployment.md) |
| RunScope | newtype | [documents](documents.md) |
| RunState | class | [deployment](deployment.md) |
| sanitize_url | func | [documents](documents.md) |
| send_spec | func | [prompt_compiler](prompt_compiler.md) |
| set_document_store | func | [document_store](document_store.md) |
| set_trace_cost | func | [observability](observability.md) |
| Settings | class | [settings](settings.md) |
| setup_logging | func | [logging](logging.md) |
| StateHookCallable | typealias | [pipeline](pipeline.md) |
| SummaryGenerator | typealias | [document_store](document_store.md) |
| SummaryUpdateFn | typealias | [document_store](document_store.md) |
| TABLE_DOCUMENT_CONTENT | constant | [document_store](document_store.md) |
| TABLE_DOCUMENT_INDEX | constant | [document_store](document_store.md) |
| TABLE_RUN_DOCUMENTS | constant | [document_store](document_store.md) |
| TARGET_ABI | constant | [deployment](deployment.md) |
| TARGET_PYTHON_VERSION | constant | [deployment](deployment.md) |
| TaskDocumentContext | class | [documents](documents.md) |
| TaskRunNameValueOrCallable | typealias | [pipeline](pipeline.md) |
| TextContent | class | [llm](llm.md) |
| TokenUsage | class | [llm](llm.md) |
| trace | func | [observability](observability.md) |
| TraceInfo | class | [observability](observability.md) |
| update | func | [deployment](deployment.md) |
| UV_TARGET_PLATFORM | constant | [deployment](deployment.md) |

## Task-Based Lookup

| Task | Guide |
| ---- | ----- |
| Create/read documents | [documents](documents.md) |
| Store/retrieve documents | [document_store](document_store.md) |
| Call LLMs | [llm](llm.md) |
| Deploy pipelines | [deployment](deployment.md) |
| Load templates | [prompt_manager](prompt_manager.md) |
| Define flows/tasks | [pipeline](pipeline.md) |
| Configure settings | [settings](settings.md) |
| Handle errors | [exceptions](exceptions.md) |
| Log messages | [logging](logging.md) |
| Debug & observe traces | [observability](observability.md) |
| Build prompts | [prompt_compiler](prompt_compiler.md) |

## Module Sizes

| Module | Size |
| ------ | ---- |
| deployment | 36,146 bytes |
| document_store | 27,703 bytes |
| documents | 40,265 bytes |
| exceptions | 1,124 bytes |
| llm | 34,326 bytes |
| logging | 8,569 bytes |
| observability | 23,224 bytes |
| pipeline | 33,593 bytes |
| prompt_compiler | 30,250 bytes |
| prompt_manager | 12,608 bytes |
| settings | 7,498 bytes |
| **Total** | **255,306 bytes** |
