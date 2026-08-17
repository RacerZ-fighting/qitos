# Examples

`examples/` is canonical learning material, not a product showcase.

The learning path is:

```text
Agent(model, tool_registry) -> prompt -> Model stream -> ToolCall -> ToolResult -> next turn
```

## Directory Map

- `examples/quickstart/`: the smallest runnable façade composition
- `examples/patterns/`: one design axis per example
- `examples/real/`: minimal environment smoke examples only

## Recommended First Run

```bash
export OPENAI_API_KEY="your_api_key"
export OPENAI_BASE_URL="https://api.openai.com/v1"
python examples/quickstart/minimal_agent.py
python examples/patterns/function_tool_custom.py
python examples/real/desktop_env_smoke.py
```

## Examples Policy

- One concept per file.
- No heavy hidden dependencies.
- No local absolute paths.
- No product clone as a canonical example.
- Security-sensitive workflows are opt-in and not part of the quickstart.
