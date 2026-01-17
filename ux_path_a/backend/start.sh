#!/bin/bash
set -e

# Ensure all output is unbuffered
export PYTHONUNBUFFERED=1

# Redirect all output to stderr so Railway captures it
exec 1>&2

VENV_PY="/app/.venv/bin/python"
if [ -x "$VENV_PY" ]; then
    PYTHON_BIN="$VENV_PY"
else
    PYTHON_BIN="python"
fi

echo "=== Starting container ===" >&2
echo "Working directory: $(pwd)" >&2
echo "Python path: $PYTHONPATH" >&2
echo "PORT: ${PORT:-8000}" >&2
echo "Python version: $($PYTHON_BIN --version)" >&2
echo "Python executable: $($PYTHON_BIN -c 'import sys; print(sys.executable)')" >&2

# Test if we can import main module
echo "Testing imports..." >&2
"$PYTHON_BIN" -c "import main; print('✓ main module imported successfully')" || {
    echo "✗ Failed to import main module" >&2
    "$PYTHON_BIN" -c "import sys; sys.path.insert(0, '.'); import main" 2>&1 || {
        echo "✗ Import failed even with path adjustment" >&2
        exit 1
    }
}

# DB + Alembic: restored by default, but run best-effort in the background
# so the web server can bind to $PORT immediately (prevents Railway fallback 502).

RUN_DB_INIT="${RUN_DB_INIT:-true}"
RUN_MIGRATIONS="${RUN_MIGRATIONS:-true}"

run_db_tasks() {
    if [ "$RUN_DB_INIT" = "true" ] || [ "$RUN_DB_INIT" = "1" ]; then
        echo "Creating base database tables (best-effort)..." >&2
        "$PYTHON_BIN" << 'PYTHON_SCRIPT' 2>&1
import traceback
try:
    from ux_path_a.backend.backend_core.database import Base, engine
    from ux_path_a.backend.backend_core.models import User, ChatSession, ChatMessage, AuditLog, TokenBudget
    Base.metadata.create_all(bind=engine)
    print("✓ Base tables created/verified")
except Exception as e:
    print(f"WARNING: Base table creation skipped/failed: {e}")
    traceback.print_exc()
PYTHON_SCRIPT
    else
        echo "Skipping DB init (set RUN_DB_INIT=false to disable)..." >&2
    fi

    if [ "$RUN_MIGRATIONS" = "true" ] || [ "$RUN_MIGRATIONS" = "1" ]; then
        echo "Running database migrations (best-effort)..." >&2
        "$PYTHON_BIN" -m alembic upgrade head 2>&1 || {
            echo "WARNING: Migration failed (best-effort)..." >&2
        }
    else
        echo "Skipping migrations (set RUN_MIGRATIONS=false to disable)..." >&2
    fi
}

# Kick off DB tasks in the background, then start the server immediately.
run_db_tasks &

# Start the server (this must succeed)
echo "Starting server on port ${PORT:-8000}..." >&2
exec "$PYTHON_BIN" -m uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000} --log-level info
