"""Custom @function_tool example.

Demonstrates creating custom tools with the @function_tool decorator,
including needs_approval, read_only, and timeout_s options,
and registering them with ToolRegistry.
"""

from qitos.core.function_tool_decorator import function_tool
from qitos.core.tool_registry import ToolRegistry


# --- Basic custom tool -------------------------------------------------------

@function_tool
def greet(name: str) -> str:
    """Greet someone by name."""
    return f"Hello, {name}!"


# --- Tool with needs_approval (requires user confirmation before execution) --

@function_tool(needs_approval=True)
def delete_resource(resource_id: str) -> str:
    """Delete a resource by ID. Requires user approval before execution."""
    return f"Deleted resource {resource_id}"


# --- Read-only tool (no side effects, safe to call freely) -------------------

@function_tool(read_only=True)
def lookup_config(key: str) -> str:
    """Look up a configuration value by key. Read-only, no side effects."""
    return f"config[{key}] = default_value"


# --- Tool with timeout (limits execution time) -------------------------------

@function_tool(timeout_s=5.0)
def slow_computation(n: int) -> int:
    """Run a computation that must complete within 5 seconds."""
    return sum(range(n))


# --- Register with ToolRegistry ----------------------------------------------

registry = ToolRegistry()
registry.register(greet)
registry.register(delete_resource)
registry.register(lookup_config)
registry.register(slow_computation)

# List all registered tools
print("Registered tools:", registry.list_tools())

# Resolve the canonical name, then execute through the tool contract
greet_tool = registry.get("greet")
assert greet_tool is not None
result = greet_tool.execute({"name": "QitOS"})
print(f"greet result: {result}")

# Check tool metadata
tool = registry.get("delete_resource")
print(f"delete_resource needs_approval: {tool.meta.needs_approval}")

tool = registry.get("lookup_config")
print(f"lookup_config read_only: {tool.meta.read_only}")

tool = registry.get("slow_computation")
print(f"slow_computation timeout_s: {tool.meta.timeout_s}")
