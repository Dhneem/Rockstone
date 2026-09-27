"""Shared pytest fixtures.

Tk on Windows cannot reliably re-initialize after all root windows have
been destroyed (``Can't find a usable init.tcl``), so GUI tests share a
single session-scoped withdrawn root instead of creating their own.
"""

from __future__ import annotations

import pytest


@pytest.fixture(scope="session")
def tk_root():
    tk = pytest.importorskip("tkinter")
    try:
        root = tk.Tk()
    except Exception as exc:  # noqa: BLE001 - headless machines
        pytest.skip(f"Tk is not available here: {exc}")
    root.withdraw()
    yield root
    try:
        root.destroy()
    except Exception:  # noqa: BLE001 - already gone
        pass
