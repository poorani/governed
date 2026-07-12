"""Where ``SessionState`` lives between checkpoints.

Three methods is the whole contract. Back it with Redis, Postgres, or S3 by
implementing ``StateStore``; the two shipped here cover tests (``InMemoryStore``)
and local, crash-safe persistence (``JSONFileStore``).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Protocol

from .session import SessionState

__all__ = ["InMemoryStore", "JSONFileStore", "StateStore"]


class StateStore(Protocol):
    def save(self, state: SessionState) -> None: ...
    def load(self, session_id: str) -> SessionState | None: ...
    def list_sessions(self) -> list[str]: ...


class InMemoryStore:
    """No persistence across process restarts. Default; fine for tests and demos."""

    def __init__(self) -> None:
        self._sessions: dict[str, SessionState] = {}

    def save(self, state: SessionState) -> None:
        self._sessions[state.session_id] = state

    def load(self, session_id: str) -> SessionState | None:
        return self._sessions.get(session_id)

    def list_sessions(self) -> list[str]:
        return list(self._sessions)


class JSONFileStore:
    """One JSON file per session, written atomically.

    ``save`` writes to a temp file in the same directory and ``os.replace``s it
    into place, so a crash mid-write cannot leave a half-written, unloadable
    session on disk -- the replace is atomic on POSIX and on Windows since
    Python 3.3.
    """

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def _path(self, session_id: str) -> Path:
        return self.directory / f"{session_id}.json"

    def save(self, state: SessionState) -> None:
        path = self._path(state.session_id)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(state.to_dict(), default=str, indent=2), encoding="utf-8")
        os.replace(tmp, path)

    def load(self, session_id: str) -> SessionState | None:
        path = self._path(session_id)
        if not path.exists():
            return None
        return SessionState.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def list_sessions(self) -> list[str]:
        return sorted(p.stem for p in self.directory.glob("*.json"))
