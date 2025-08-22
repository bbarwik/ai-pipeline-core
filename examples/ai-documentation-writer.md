# AI Documentation Writer - Real-World Example

[AI Documentation Writer](https://github.com/bbarwik/ai-documentation-writer) is a comprehensive example of how to build production-ready AI pipelines using `ai-pipeline-core`.

## What It Does

AI Documentation Writer automatically generates comprehensive documentation for code repositories using advanced AI analysis. It demonstrates how to build complex, multi-stage pipelines that process large codebases efficiently.

## Key Features Demonstrated

### 1. Structured Document Workflow
- Uses strongly-typed Pydantic models for each pipeline stage
- Creates self-contained, versioned documents at each processing step
- Shows proper document inheritance patterns with `FlowDocument` and `TaskDocument`

### 2. Async Architecture
- Fully asynchronous processing for optimal performance
- Parallel batch processing of files
- Demonstrates "pure async functions with full type hints" pattern

### 3. Modular Pipeline Design
The project implements a 3-stage resumable pipeline:
1. **Prepare Files** - Analyzes repository structure and prepares files for processing
2. **Generate Description** - Creates AI-powered descriptions for each code component
3. **Document Codebase** - Assembles comprehensive documentation

### 4. Advanced Patterns
- Custom FlowOptions for configuration management
- Intelligent context caching for LLM calls
- Error handling and retry strategies
- Progress tracking and resumable workflows

## Learning Points

This project showcases:
- How to structure large AI applications with `ai-pipeline-core`
- Best practices for document typing and validation
- Efficient LLM usage with context management
- Building resumable, fault-tolerant pipelines
- Integration with Prefect for orchestration

## Run It Yourself

```bash
# Clone the repository
git clone https://github.com/bbarwik/ai-documentation-writer
cd ai-documentation-writer

# Install dependencies
pip install -e .

# Run on your project
ai-doc-writer /path/to/your/project
```

This real-world example demonstrates the power and flexibility of `ai-pipeline-core` for building sophisticated AI-powered applications.
