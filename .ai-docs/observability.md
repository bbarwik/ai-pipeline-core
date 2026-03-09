# MODULE: observability
# PURPOSE: Observability system for AI pipelines.
# VERSION: 0.14.0
# AUTO-GENERATED from source code — do not edit. Run: make docs-ai-build

## Functions

```python
def main(argv: list[str] | None = None) -> int:
    """Run the ai-trace CLI."""
    parser = argparse.ArgumentParser(prog="ai-trace", description="Inspect deployment execution trees")
    subparsers = parser.add_subparsers(dest="command")

    list_parser = subparsers.add_parser("list", help="List recent deployments")
    list_parser.add_argument("--limit", type=int, default=20, help="Maximum number of deployments to show")
    list_parser.add_argument("--status", type=str, default=None, help="Filter deployments by status")
    list_parser.add_argument("--db-path", type=str, default=None, help="Use a FilesystemDatabase snapshot instead of ClickHouse")

    show_parser = subparsers.add_parser("show", help="Show deployment summary and logs")
    show_parser.add_argument("identifier", help="Deployment/node UUID or deployment run_id")
    show_parser.add_argument("--db-path", type=str, default=None, help="Use a FilesystemDatabase snapshot instead of ClickHouse")

    download_parser = subparsers.add_parser("download", help="Download a deployment as a FilesystemDatabase snapshot")
    download_parser.add_argument("identifier", help="Deployment/node UUID or deployment run_id")
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

**Download command** (`tests/observability/test_trace_cli.py:168`)

```python
def test_download_command(self, tmp_path: Path) -> None:
    _seed_trace_database(tmp_path / "source")
    output_dir = tmp_path / "download"
    result = main(["download", "run-001", "--db-path", str(tmp_path / "source"), "--output-dir", str(output_dir)])
    assert result == 0
    assert (output_dir / "summary.md").exists()
    assert (output_dir / "costs.md").exists()
    assert (output_dir / "logs.jsonl").exists()
    assert (output_dir / "runs").is_dir()
```

**List command** (`tests/observability/test_trace_cli.py:123`)

```python
def test_list_command(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _seed_trace_database(tmp_path)
    result = main(["list", "--db-path", str(tmp_path)])
    assert result == 0
    assert "trace-cli-test" in capsys.readouterr().out
```

**Show command** (`tests/observability/test_trace_cli.py:129`)

```python
def test_show_command(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _seed_trace_database(tmp_path)
    result = main(["show", "run-001", "--db-path", str(tmp_path)])
    assert result == 0
    output = capsys.readouterr().out
    assert "# trace-cli-test / run-001" in output
    assert "deployment started" in output
    assert "task finished" in output
```

**Db path returns filesystem database** (`tests/observability/test_trace_cli.py:99`)

```python
def test_db_path_returns_filesystem_database(self, tmp_path: Path) -> None:
    args = type("Args", (), {"db_path": str(tmp_path)})()
    result = _resolve_connection(args)
    assert isinstance(result, FilesystemDatabase)
```

**Run id resolves deployment** (`tests/observability/test_trace_cli.py:111`)

```python
def test_run_id_resolves_deployment(self, tmp_path: Path) -> None:
    database, deployment, _task = _seed_trace_database(tmp_path)
    result = _resolve_identifier(deployment.run_id, database)
    assert result == (deployment.deployment_id, deployment.run_id)
```

**Uuid returns deployment id and run id** (`tests/observability/test_trace_cli.py:106`)

```python
def test_uuid_returns_deployment_id_and_run_id(self, tmp_path: Path) -> None:
    database, deployment, task = _seed_trace_database(tmp_path)
    result = _resolve_identifier(str(task.node_id), database)
    assert result == (deployment.deployment_id, deployment.run_id)
```


## Error Examples

**Invalid uuid exits** (`tests/observability/test_trace_cli.py:93`)

```python
def test_invalid_uuid_exits(self) -> None:
    with pytest.raises(SystemExit):
        _parse_execution_id("not-a-uuid")
```

**Missing identifier exits** (`tests/observability/test_trace_cli.py:116`)

```python
def test_missing_identifier_exits(self, tmp_path: Path) -> None:
    database = FilesystemDatabase(tmp_path)
    with pytest.raises(SystemExit):
        _resolve_identifier("missing-run", database)
```
