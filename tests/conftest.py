"""Test-only stand-ins for the Hermes host modules.

``optchat/__init__`` imports ``agent.memory_provider`` at module load; the
real modules live in the hermes-agent checkout, which contributors may not
have on disk. When the checkout is absent, install minimal stand-ins so the
suite runs standalone. The stubs are marked with ``__test_stub__`` and are
never installed when the real ``agent`` package imports cleanly.

The provider-wiring tests (test_provider.py) still require the real ABC
and skip when only the stub is active.
"""

import os
import sys
import threading
import types
from dataclasses import dataclass, field
from typing import Any, Dict


def _install_stubs():
    try:
        import agent.memory_provider  # noqa: F401
        import hermes_constants  # noqa: F401
        return  # real checkout present; never shadow it
    except ImportError:
        pass

    agent = types.ModuleType("agent")
    mp = types.ModuleType("agent.memory_provider")
    mp.__test_stub__ = True

    class MemoryProvider:
        """Import-time stand-in for the Hermes MemoryProvider ABC."""

    @dataclass
    class RecallStatus:
        provider_label: str = ""
        count: int = 0
        extra: Dict[str, Any] = field(default_factory=dict)

    def is_core_memory_provider(name: str) -> bool:
        return False

    def spawn_context_thread(target, name=None):
        # The real one copies contextvars; plain daemon thread is enough here.
        return threading.Thread(target=target, name=name, daemon=True)

    mp.MemoryProvider = MemoryProvider
    mp.RecallStatus = RecallStatus
    mp.is_core_memory_provider = is_core_memory_provider
    mp.spawn_context_thread = spawn_context_thread

    hc = types.ModuleType("hermes_constants")

    def hermes_home_key(hermes_home=None):
        h = hermes_home or os.environ.get("HERMES_HOME", "")
        return os.path.abspath(os.path.expanduser(h))

    hc.hermes_home_key = hermes_home_key

    agent.memory_provider = mp
    sys.modules["agent"] = agent
    sys.modules["agent.memory_provider"] = mp
    sys.modules["hermes_constants"] = hc


_install_stubs()
