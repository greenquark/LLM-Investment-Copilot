#!/bin/bash
set -e

# Ensure all output is unbuffered
export PYTHONUNBUFFERED=1

# Redirect all output to stderr so Railway captures it
exec 1>&2

echo "=== Starting container ===" >&2
echo "Working directory: $(pwd)" >&2
echo "Python path: $PYTHONPATH" >&2
echo "PORT: ${PORT:-8000}" >&2
echo "Python version: $(python --version)" >&2

# Test if we can import main module
echo "Testing imports..." >&2
python -c "import main; print('✓ main module imported successfully')" || {
    echo "✗ Failed to import main module" >&2
    python -c "import sys; sys.path.insert(0, '.'); import main" 2>&1 || {
        echo "✗ Import failed even with path adjustment" >&2
        exit 1
    }
}

# IMPORTANT: Do not block server startup on DB operations.
# Railway Edge will return fallback 502 if the process hasn't bound to $PORT yet.
# If you want to run DB init/migrations, enable them explicitly via env vars.

RUN_DB_INIT="${RUN_DB_INIT:-false}"
if [ "$RUN_DB_INIT" = "true" ] || [ "$RUN_DB_INIT" = "1" ]; then
    echo "Creating base database tables (best-effort)..." >&2
    set +e
    python << 'PYTHON_SCRIPT' 2>&1
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
    status=$?
    set -e
    if [ $status -ne 0 ]; then
        echo "WARNING: Base table creation returned non-zero ($status), continuing..." >&2
    fi
else
    echo "Skipping DB init (set RUN_DB_INIT=true to enable)..." >&2
fi

RUN_MIGRATIONS="${RUN_MIGRATIONS:-false}"
if [ "$RUN_MIGRATIONS" = "true" ] || [ "$RUN_MIGRATIONS" = "1" ]; then
    echo "Running database migrations (best-effort)..." >&2
    python -m alembic upgrade head 2>&1 || {
        echo "WARNING: Migration failed, continuing to start server..." >&2
    }
else
    echo "Skipping migrations (set RUN_MIGRATIONS=true to enable)..." >&2
fi

# Start the server (this must succeed)
echo "Starting server on port ${PORT:-8000}..." >&2
exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000} --log-level info
