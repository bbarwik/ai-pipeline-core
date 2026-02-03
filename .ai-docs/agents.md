# MODULE: agents
# CLASSES: AgentResult, AgentProvider, AgentOutputDocument
# DEPENDS: ABC, BaseModel
# SIZE: ~17KB

# === DEPENDENCIES (Resolved) ===

class ABC:
    """Python abstract base class marker."""
    ...

class BaseModel:
    """Pydantic base model. Fields are typed class attributes."""
    ...

# === PUBLIC API ===

@dataclass(frozen=True)
class AgentResult:
    """Result from agent execution.

This is an immutable container for all data returned by an agent run.
The frozen=True ensures results cannot be accidentally modified.

Attributes:
    success: Whether the agent completed successfully
    output: Structured output data from the agent (dict)
    artifacts: Binary files produced by the agent {name: bytes}
    error: Error message if success=False
    traceback: Full traceback if success=False
    exit_code: Process exit code if applicable
    duration_seconds: Total execution time
    stdout: Captured stdout if available
    stderr: Captured stderr if available
    agent_name: Name of the agent that ran
    agent_version: Version of the agent"""
    success: bool
    output: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, bytes] = field(default_factory=dict)
    error: str | None = None
    traceback: str | None = None
    exit_code: int | None = None
    duration_seconds: float | None = None
    stdout: str | None = None
    stderr: str | None = None
    agent_name: str | None = None
    agent_version: str | None = None

    def get_artifact(self, name: str, encoding: str = "utf-8") -> str | None:
        """Get artifact content decoded as string.

        Args:
            name: Artifact filename (e.g., "output.md")
            encoding: Text encoding (default UTF-8)

        Returns:
            Decoded string content, or None if artifact doesn't exist
        """
        data = self.artifacts.get(name)
        if data is not None:
            return data.decode(encoding)
        return None

    def get_artifact_bytes(self, name: str) -> bytes | None:
        """Get artifact content as raw bytes.

        Args:
            name: Artifact filename

        Returns:
            Raw bytes, or None if artifact doesn't exist
        """
        return self.artifacts.get(name)


class AgentProvider(ABC):
    """Abstract base class for agent execution providers.

Implementations handle all details of agent execution:
- Resolving agent name to executable (local path, downloaded bundle, etc.)
- Executing the agent (subprocess, remote worker, container, etc.)
- Collecting results and artifacts

ai-pipeline-core only interacts through this interface."""
    async def list_agents(self) -> list[str]:
        """List available agent names.

        Override this method to support agent discovery.
        Default implementation returns empty list.

        Returns:
            List of agent names that can be passed to run()
        """
        return []

    @abstractmethod
    async def run(
        self,
        agent_name: str,
        inputs: dict[str, Any],
        *,
        files: dict[str, str | bytes] | None = None,
        target_worker: str | None = None,
        backend: str | None = None,
        timeout_seconds: int = 3600,
        env_vars: dict[str, str] | None = None,
    ) -> AgentResult:
        """Execute an agent and return results.

        This is the main entry point for running agents. Implementations
        should handle resolution, execution, and result collection.

        Args:
            agent_name: Identifier for the agent (e.g., "initial_research")
            inputs: Input parameters passed to the agent
            files: Additional files to make available to the agent
            target_worker: Hint for which worker to run on (provider-specific)
            backend: Which backend to use (provider-specific, e.g., "codex")
            timeout_seconds: Maximum execution time before cancellation
            env_vars: Environment variables to set for the agent

        Returns:
            AgentResult containing success status, outputs, and artifacts

        Raises:
            ValueError: If agent_name is invalid
            RuntimeError: If provider is misconfigured
        """

    async def validate(self) -> None:
        """Validate provider configuration.

        Override this method to add configuration checks.
        Called lazily on first run() to fail fast with clear errors.

        Raises:
            RuntimeError: If provider is misconfigured with helpful message
        """


class AgentOutputDocument(Document):
    """Document wrapping agent execution results.

Use this class when creating @pipeline_task functions that run agents.
It converts AgentResult to a Document that:
- Contains the primary artifact as content
- Optionally includes other artifacts as attachments
- Tracks provenance via origins parameter

Usage: See tests/agents/test_documents.py for usage patterns."""
    # [Inherited from Document]
    # __init__, __init_subclass__, approximate_tokens_count, as_json, as_pydantic_model, as_yaml, canonical_name, content_sha256, create, from_dict, get_expected_files, has_source, id, is_image, is_pdf, is_text, mime_type, model_convert, parse, serialize_content, serialize_model, sha256, size, source_documents, source_references, text, validate_content, validate_file_name, validate_name, validate_no_source_origin_overlap, validate_origins, validate_sources, validate_total_size

    @classmethod
    def from_result(
        cls,
        result: AgentResult,
        *,
        artifact_name: str = "output.md",
        name: str = "agent_output.md",
        description: str = "",
        origins: tuple[str, ...],  # Required! Enforces provenance
        include_artifacts_as_attachments: bool = False,
        max_attachment_size: int = 10 * 1024 * 1024,  # 10MB
    ) -> "AgentOutputDocument":
        """Create document from agent result.

        Args:
            result: AgentResult from run_agent()
            artifact_name: Name of artifact to use as primary content
            name: Document name (filename)
            description: Human-readable description
            origins: SHA256 hashes of documents that caused this execution.
                     REQUIRED - agents don't run in isolation, something
                     triggered them. Pass the triggering document's SHA256.
            include_artifacts_as_attachments: If True, include other artifacts
                as document attachments
            max_attachment_size: Skip attachments larger than this (bytes)

        Returns:
            AgentOutputDocument instance

        Raises:
            ValueError: If origins is empty (provenance required)
        """
        if not origins:
            raise ValueError(
                "origins is required for AgentOutputDocument. "
                "Pass the SHA256 of the document(s) that triggered this agent run. "
                "Example: origins=(input_doc.sha256,)"
            )

        # Extract primary content
        if result.success:
            content = result.get_artifact(artifact_name) or ""
            if not content:
                # Fallback: find any markdown file
                for art_name in result.artifacts:
                    if art_name.endswith(".md"):
                        content = result.get_artifact(art_name) or ""
                        break
                if not content:
                    content = f"Agent completed but no output artifact found.\nArtifacts: {list(result.artifacts.keys())}"
        else:
            # Build error content with all available info
            parts = [f"Agent failed: {result.error or 'Unknown error'}"]
            if result.traceback:
                parts.append(f"\n\n## Traceback\n```\n{result.traceback}\n```")
            if result.stderr:
                parts.append(f"\n\n## Stderr\n```\n{result.stderr}\n```")
            if result.exit_code is not None:
                parts.append(f"\n\nExit code: {result.exit_code}")
            content = "".join(parts)

        # Build attachments from other artifacts (optional)
        attachments: list[Attachment] | None = None
        if include_artifacts_as_attachments and result.artifacts:
            attachments = []
            for art_name, art_bytes in result.artifacts.items():
                # Skip primary content artifact
                if art_name == artifact_name:
                    continue

                # Skip large files
                if len(art_bytes) > max_attachment_size:
                    logger.warning(f"Skipping large artifact: {art_name} ({len(art_bytes)} > {max_attachment_size} bytes)")
                    continue

                # Sanitize name
                safe_name = _sanitize_name(art_name)
                if safe_name is None:
                    logger.warning(f"Skipping artifact with invalid name: {art_name}")
                    continue

                attachments.append(
                    Attachment(
                        name=safe_name,
                        content=art_bytes,
                        description=f"Agent artifact: {art_name}",
                    )
                )

        return cls.create(
            name=name,
            content=content,
            description=description or f"Output from agent: {result.agent_name or 'unknown'}",
            origins=origins,
            attachments=tuple(attachments) if attachments else (),
        )


# === FUNCTIONS ===

def register_agent_provider(provider: AgentProvider) -> None:
    """Register an agent provider.

    Call once at application startup, typically in __init__.py.
    Raises if a provider is already registered to prevent accidental
    overwrites. Use reset_agent_provider() first if you need to replace.

    Args:
        provider: The AgentProvider implementation to register

    Raises:
        RuntimeError: If a provider is already registered
    """
    global _provider
    with _lock:
        if _provider is not None:
            raise RuntimeError(f"Agent provider already registered: {type(_provider).__name__}. Call reset_agent_provider() first to replace it.")
        _provider = provider

def get_agent_provider() -> AgentProvider:
    """Get the registered agent provider.

    Returns:
        The registered AgentProvider instance

    Raises:
        RuntimeError: If no provider is registered, with instructions
            on how to register one (generic, no specific package names)
    """
    with _lock:
        if _provider is None:
            raise RuntimeError(
                "No agent provider registered.\n\n"
                "To use agents, install an agent provider package and register it.\n"
                "See tests/agents/conftest.py for example of provider registration."
            )
        return _provider

def reset_agent_provider() -> None:
    """Reset the provider registration.

    Primarily for testing - allows registering a new provider.
    In production code, providers should be registered once at startup.
    """
    global _provider
    with _lock:
        _provider = None

@contextmanager
def temporary_provider(provider: AgentProvider) -> Iterator[AgentProvider]:
    """Context manager for temporarily registering a provider.

    Useful for testing - registers the provider, yields it, then
    restores the previous registration on exit (even if an exception occurs).

    Args:
        provider: The provider to temporarily register

    Yields:
        The registered provider
    """
    global _provider
    with _lock:
        previous_provider = _provider
        _provider = provider
    try:
        yield provider
    finally:
        with _lock:
            _provider = previous_provider

async def run_agent(
    agent_name: str,
    inputs: dict[str, Any],
    *,
    files: dict[str, str | bytes] | None = None,
    target_worker: str | None = None,
    backend: str | None = None,
    timeout_seconds: int = 3600,
    env_vars: dict[str, str] | None = None,
) -> AgentResult:
    """Execute an agent via the registered provider.

    This function is the main entry point for running agents. It:
    1. Gets the registered provider
    2. Validates the provider on first use (lazy validation)
    3. Delegates to provider.run()
    4. Logs success/failure

    The actual execution logic is entirely in the provider implementation.
    This keeps ai-pipeline-core decoupled from any specific agent system.

    Args:
        agent_name: Name of the agent to run (e.g., "initial_research")
        inputs: Input parameters passed to the agent as a dict
        files: Additional files to include in agent workspace {name: content}
        target_worker: Which worker to run on (provider-specific)
        backend: Which backend to use (provider-specific, e.g., "codex")
        timeout_seconds: Maximum execution time (default 1 hour)
        env_vars: Environment variables to set for the agent

    Returns:
        AgentResult with success status, output, and artifacts

    Raises:
        RuntimeError: If no provider is registered
        ValueError: If agent_name is invalid
    """
    provider = get_agent_provider()
    provider_name = type(provider).__name__
    provider_id = id(provider)

    logger.info(f"Running agent '{agent_name}' via {provider_name}")

    # Lazy validation on first use (with lock to prevent duplicate validation)
    # Using id() to track validated providers - works with all providers including __slots__
    if provider_id not in _validated_provider_ids:
        async with _validation_lock:
            # Double-check after acquiring lock
            if provider_id not in _validated_provider_ids:
                await provider.validate()
                _validated_provider_ids.add(provider_id)

    result = await provider.run(
        agent_name=agent_name,
        inputs=inputs,
        files=files,
        target_worker=target_worker,
        backend=backend,
        timeout_seconds=timeout_seconds,
        env_vars=env_vars,
    )

    if result.success:
        duration = f"{result.duration_seconds:.1f}s" if result.duration_seconds else "unknown"
        logger.info(f"Agent '{agent_name}' completed successfully (duration: {duration})")
    else:
        logger.warning(f"Agent '{agent_name}' failed: {result.error}")

    return result

# === EXAMPLES (from tests/) ===

# Example: Nested temporary providers
# Source: tests/agents/test_registry.py:133
def test_nested_temporary_providers(self):
    """Nested temporary_provider() should work correctly."""
    provider_a = MockAgentProvider()
    provider_b = MockAgentProvider()
    provider_c = MockAgentProvider()

    register_agent_provider(provider_a)

    with temporary_provider(provider_b):
        assert get_agent_provider() is provider_b

        with temporary_provider(provider_c):
            assert get_agent_provider() is provider_c

        assert get_agent_provider() is provider_b

    assert get_agent_provider() is provider_a

# Example: Error shows existing provider type
# Source: tests/agents/test_registry.py:55
def test_error_shows_existing_provider_type(self, mock_provider: MockAgentProvider):
    """Error should mention the type of already-registered provider."""
    register_agent_provider(mock_provider)

    try:
        register_agent_provider(MockAgentProvider())
        pytest.fail("Should have raised RuntimeError")
    except RuntimeError as e:
        assert "MockAgentProvider" in str(e)

# Example: Logs provider name
# Source: tests/agents/test_runner.py:197
async def test_logs_provider_name(self, mock_provider: MockAgentProvider, caplog):
    """Should log which provider is being used."""
    register_agent_provider(mock_provider)

    with caplog.at_level("INFO"):
        await run_agent("my_agent", {})

    assert "MockAgentProvider" in caplog.text
    assert "my_agent" in caplog.text

# Example: Minimal arguments
# Source: tests/agents/test_runner.py:59
async def test_minimal_arguments(self, mock_provider: MockAgentProvider):
    """run_agent() should work with minimal arguments."""
    register_agent_provider(mock_provider)

    result = await run_agent("agent_name", {"key": "value"})

    assert result.success
    assert len(mock_provider.run_calls) == 1

# === ERROR EXAMPLES (What NOT to Do) ===

# Error: Get without registration raises
# Source: tests/agents/test_registry.py:29
def test_get_without_registration_raises(self):
    """get_agent_provider() should raise with helpful message when nothing registered."""
    with pytest.raises(RuntimeError, match="No agent provider registered"):
        get_agent_provider()

# Error: Is abstract
# Source: tests/agents/test_base.py:91
def test_is_abstract(self):
    """AgentProvider cannot be instantiated directly."""
    with pytest.raises(TypeError, match="abstract"):
        AgentProvider()  # type: ignore

# Error: Is frozen
# Source: tests/agents/test_base.py:12
def test_is_frozen(self, sample_result: AgentResult):
    """AgentResult should be immutable."""
    with pytest.raises(FrozenInstanceError):
        sample_result.success = False  # type: ignore

# Error: Register twice raises
# Source: tests/agents/test_registry.py:47
def test_register_twice_raises(self, mock_provider: MockAgentProvider):
    """Registering a second provider without reset should raise."""
    register_agent_provider(mock_provider)

    another_provider = MockAgentProvider()
    with pytest.raises(RuntimeError, match="already registered"):
        register_agent_provider(another_provider)
