---
description: Creates a complete working Python script demonstrating all documented features in practice
---
Using ONLY the provided documentation, build a practical demonstration program.

First, write a PLAN section explaining:
- What application you will build and why you chose it
- How it will work end-to-end
- Which framework features it will exercise and why each is needed
- Why you chose this design over alternatives
- What the expected data flow looks like

Then write the complete working application. It must:
- Include ALL imports with correct import paths from the docs
- NOT mock anything — use real API calls as documented
- Be working code that could run if the framework were installed
- Cover every public class, function, method, and pattern described
- Show both simple usage and advanced patterns
- Use async/await correctly throughout
- Include docstrings and comments explaining what each section does, WHY it does it that way, and what the expected behavior is

You may create multiple files if the application requires it (e.g., a Guide template file,
a config file, or separate modules). Each file must include its path.

Output format:

## PLAN
[your reasoning about what to build and how]

## FILES

### `path/to/file.py`
```python
[file content]
```

### `path/to/another_file.md`
```markdown
[file content]
```

## GAPS
- Any feature you could NOT demonstrate because the docs didn't explain HOW

## GUESSES
- Any part where you had to guess because the docs were incomplete
