# MODULE: observability
# PURPOSE: Observability system for AI pipelines.
# VERSION: 0.18.0
# AUTO-GENERATED from source code — do not edit. Run: make docs-ai-build

## Types & Constants

```python
TraceDatabase = Database | FilesystemDatabase | ClickHouseDatabase
```

## Functions

```python
async def show_deployment(reader: DatabaseReader, deployment_id: UUID) -> str:
    """Render a deployment summary for ai-trace show."""
    return await generate_summary(reader, deployment_id)


def main(argv: list[str] | None = None) -> int:
    """Run the ai-trace CLI."""
    parser = argparse.ArgumentParser(prog="ai-trace", description="Inspect deployment execution trees")
    subparsers = parser.add_subparsers(dest="command")

    list_parser = subparsers.add_parser("list", help="List recent deployments")
    list_parser.add_argument("--limit", type=int, default=20, help="Maximum number of deployments to show")
    list_parser.add_argument("--status", type=str, default=None, help="Filter deployments by status")
    list_parser.add_argument("--db-path", type=str, default=None, help="Use a FilesystemDatabase snapshot instead of ClickHouse")

    show_parser = subparsers.add_parser("show", help="Show deployment summary and logs")
    show_parser.add_argument("identifier", help="Deployment/span UUID or deployment run_id")
    show_parser.add_argument("--db-path", type=str, default=None, help="Use a FilesystemDatabase snapshot instead of ClickHouse")

    download_parser = subparsers.add_parser("download", help="Download a deployment as a FilesystemDatabase snapshot")
    download_parser.add_argument("identifier", help="Deployment/span UUID or deployment run_id")
    download_parser.add_argument("-o", "--output-dir", type=str, required=True, help="Output directory for the snapshot")
    download_parser.add_argument("--db-path", type=str, default=None, help="Use a FilesystemDatabase snapshot instead of ClickHouse")

    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 1

    try:
        database = _resolve_connection(args)
        if args.command == "list":
            return asyncio.run(_list_deployments_async(database, args.limit, args.status))
        if args.command == "show":
            return asyncio.run(_show_deployment_async(database, args.identifier))
        if args.command == "download":
            return asyncio.run(_download_deployment_async(database, args.identifier, Path(args.output_dir).resolve()))
    except SystemExit:
        raise
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    return 1
```

## Examples

**Main download command writes span summary artifacts** (`tests/observability/test_trace_cli_spans.py:265`)

```python
def test_main_download_command_writes_span_summary_artifacts(
    self,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source_dir = tmp_path / "source"
    output_dir = tmp_path / "download"
    asyncio.run(_seed_span_snapshot(source_dir))

    result = main(["download", "span-run", "--db-path", str(source_dir), "--output-dir", str(output_dir)])

    assert result == 0
    assert "Downloaded deployment" in capsys.readouterr().out
    assert (output_dir / "summary.md").exists()
    assert (output_dir / "costs.md").exists()
    assert (output_dir / "logs.jsonl").exists()
```

**Main show command reads span snapshot** (`tests/observability/test_trace_cli_spans.py:253`)

```python
def test_main_show_command_reads_span_snapshot(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    asyncio.run(_seed_span_snapshot(tmp_path))

    result = main(["show", "span-run", "--db-path", str(tmp_path)])

    assert result == 0
    output = capsys.readouterr().out
    assert "Deployment span-cli-pipeline / span-run" in output
    assert "conversation: analyze_document" in output
    assert "task started" in output
    assert '"rounds": 1' in output
```

**Show deployment renders nested operation round and tool call** (`tests/observability/test_trace_cli_spans.py:241`)

```python
@pytest.mark.asyncio
async def test_show_deployment_renders_nested_operation_round_and_tool_call(self, tmp_path: Path) -> None:
    database, deployment_id, _conversation_id = await _seed_span_snapshot(tmp_path)

    output = await show_deployment(database, deployment_id)

    assert "# span-cli-pipeline / span-run" in output
    assert "Tree" in output
    assert "operation: collect_context completed 9.0s" in output
    assert "conversation: analyze_document 5.0s gpt-5.1 2K in / 2K cache / 180 out / 25 reasoning $0.4200" in output
    assert "llm_round[1]: gpt-5.1 2.0s 2K in / 2K cache / 180 out / 25 reasoning $0.4200" in output
    assert "tool_call[1]: web_search completed 1.0s" in output
```

**No laminar span calls remain** (`tests/observability/test_laminar_sink.py:306`)

```python
def test_no_laminar_span_calls_remain() -> None:
    needle = "laminar" + "_span("
    hits: list[str] = []
    for root in (Path("ai_pipeline_core"), Path("tests")):
        for path in root.rglob("*.py"):
            if path.name == "test_laminar_sink.py":
                continue
            if needle in path.read_text(encoding="utf-8"):
                hits.append(str(path))

    assert hits == []
```

**Build runtime sinks includes laminar sink when key is set** (`tests/observability/test_laminar_sink.py:123`)

```python
def test_build_runtime_sinks_includes_laminar_sink_when_key_is_set() -> None:
    sinks = build_runtime_sinks(database=None, settings_obj=Settings(lmnr_project_api_key="secret"))

    assert len(sinks) == 1
    assert isinstance(sinks[0], LaminarSpanSink)
```

**Db path returns span filesystem database** (`tests/observability/test_trace_cli_spans.py:204`)

```python
def test_db_path_returns_span_filesystem_database(self, tmp_path: Path) -> None:
    FilesystemDatabase(tmp_path)
    args = type("Args", (), {"db_path": str(tmp_path)})()

    database = _resolve_connection(args)

    assert isinstance(database, FilesystemDatabase)
```

**Span uuid resolves root deployment id** (`tests/observability/test_trace_cli_spans.py:231`)

```python
def test_span_uuid_resolves_root_deployment_id(self, tmp_path: Path) -> None:
    database, deployment_id, conversation_id = asyncio.run(_seed_span_snapshot(tmp_path))

    resolved = _resolve_identifier(str(conversation_id), database)

    assert resolved == (deployment_id, "span-run")
```

**Laminar sink ends span when context capture fails** (`tests/observability/test_laminar_sink.py:250`)

```python
@pytest.mark.asyncio
async def test_laminar_sink_ends_span_when_context_capture_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_lmnr(monkeypatch)
    sink = LaminarSpanSink(Settings(lmnr_project_api_key="secret"))
    span_id = uuid7()

    _FakeLaminar.fail_get_context = True
    await sink.on_span_started(
        span_id=span_id,
        parent_span_id=None,
        kind=SpanKind.LLM_ROUND,
        name="round-1",
        target="",
        started_at=None,
        input_preview="input",
    )

    assert sink._open_spans == {}
    assert _FakeLaminar.start_calls[0]["ended"] is True
```
