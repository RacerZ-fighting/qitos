"""Stream transformer protocol and built-in transformers.

StreamTransformers customize the output of Engine.arun_stream()
by filtering, projecting, or converting EngineEvents into different
output formats.

Usage::

    from qitos.engine.stream import ValuesTransformer, MessagesTransformer

    async for output in engine.arun_stream(task, transformers=[ValuesTransformer()]):
        print(output)
"""

from .transformer import StreamTransformer, TransformerOutput
from .values import ValuesTransformer
from .messages import MessagesTransformer
from .lifecycle import LifecycleTransformer

__all__ = [
    "StreamTransformer",
    "TransformerOutput",
    "ValuesTransformer",
    "MessagesTransformer",
    "LifecycleTransformer",
]
