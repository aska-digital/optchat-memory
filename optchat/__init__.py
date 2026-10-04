"""optchat-memory: OptChat-style endless memory for Hermes Agent.

Standalone memory-provider plugin. Install via the Hermes plugin catalog
(``hermes plugins install optchat-memory``) or drop this package directory
into ``$HERMES_HOME/plugins/optchat/``. Activate with
``memory.provider: optchat`` (``hermes memory setup``), one provider at a time.
"""

from .provider import OptChatMemoryProvider, PROVIDER_NAME

__all__ = ["OptChatMemoryProvider", "PROVIDER_NAME", "register"]
__version__ = "0.1.0"


def register(ctx) -> None:
    """Register optchat as a memory provider plugin."""
    ctx.register_memory_provider(OptChatMemoryProvider())
