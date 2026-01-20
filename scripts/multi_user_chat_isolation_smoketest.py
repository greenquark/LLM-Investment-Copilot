"""
Smoke test: ensure chat sessions/messages are isolated per user.

This catches a critical authz bug where /api/chat/sessions returned all sessions
and message endpoints didn't enforce ownership.

Runs entirely in-process via FastAPI TestClient against a temporary SQLite DB.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path


def _post_form(client, path: str, data: dict, headers: dict | None = None):
    return client.post(path, data=data, headers=headers or {})


def _auth_headers(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def main() -> int:
    # Ensure repo root is on sys.path so namespace package imports like `ux_path_a.*` work
    # when running this file from the `scripts/` directory.
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    # IMPORTANT: set env BEFORE importing the app so settings/engine pick it up.
    tmp_dir = Path(tempfile.mkdtemp(prefix="ux_path_a_multiuser_smoke_"))
    db_path = tmp_dir / "ux_path_a_multiuser_smoke.db"
    os.environ["DATABASE_URL"] = f"sqlite:///{db_path.as_posix()}"
    os.environ["RUN_DB_STARTUP"] = "true"
    # Ensure we don't hit the real OpenAI API in this smoke test; we only verify authz.
    os.environ.pop("OPENAI_API_KEY", None)

    from fastapi.testclient import TestClient  # noqa: WPS433 (runtime import)
    from ux_path_a.backend.main import app  # noqa: WPS433 (runtime import)

    # Use a context manager so FastAPI startup runs (creates tables).
    with TestClient(app) as client:
        # Login as two distinct users.
        r1 = _post_form(client, "/api/auth/token", {"username": "alice", "password": "pw"})
        assert r1.status_code == 200, r1.text
        token_alice = r1.json()["access_token"]

        r2 = _post_form(client, "/api/auth/token", {"username": "bob", "password": "pw"})
        assert r2.status_code == 200, r2.text
        token_bob = r2.json()["access_token"]

        # Alice creates a session.
        r = client.post("/api/chat/sessions", json={"title": "Alice chat"}, headers=_auth_headers(token_alice))
        assert r.status_code == 200, r.text
        alice_session_id = r.json()["id"]

        # Alice sees her session; Bob does not.
        r = client.get("/api/chat/sessions", headers=_auth_headers(token_alice))
        assert r.status_code == 200, r.text
        alice_sessions = r.json()
        assert any(s["id"] == alice_session_id for s in alice_sessions), alice_sessions

        r = client.get("/api/chat/sessions", headers=_auth_headers(token_bob))
        assert r.status_code == 200, r.text
        bob_sessions = r.json()
        assert all(s["id"] != alice_session_id for s in bob_sessions), bob_sessions

        # Bob cannot read Alice's messages.
        r = client.get(f"/api/chat/sessions/{alice_session_id}/messages", headers=_auth_headers(token_bob))
        assert r.status_code == 404, r.text

        # Bob cannot post into Alice's session. (Should be blocked before LLM config checks.)
        r = client.post(
            "/api/chat/messages",
            json={"content": "hi", "session_id": alice_session_id},
            headers=_auth_headers(token_bob),
        )
        assert r.status_code == 404, r.text

    print("OK: chat sessions/messages are isolated per user.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

