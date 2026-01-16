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

# Create base tables first (if they don't exist)
echo "Creating base database tables..." >&2
set +e
python << 'PYTHON_SCRIPT' 2>&1
import sys
import traceback

try:
    print("Attempting to import database modules...")
    # Use absolute imports (works in both local and Railway with PYTHONPATH=/app)
    from ux_path_a.backend.backend_core.database import Base, engine
    from ux_path_a.backend.backend_core.models import User, ChatSession, ChatMessage, AuditLog, TokenBudget
    print("✓ Imported using absolute imports")
    print("Creating tables from models (best-effort)...")
    Base.metadata.create_all(bind=engine)
    print("✓ Base tables created/verified")
except Exception as e:
    # Best-effort only: never block server startup on DB/table creation.
    # Health endpoints should still be able to come up even if DB is temporarily unavailable.
    print(f"WARNING: Base table creation skipped/failed: {e}")
    traceback.print_exc()
    sys.exit(0)
PYTHON_SCRIPT
status=$?
set -e

if [ $status -ne 0 ]; then
    echo "WARNING: Base table creation returned non-zero ($status), but continuing..." >&2
fi

# Run database migrations (don't fail if migrations error)
echo "Running database migrations..." >&2
python -m alembic upgrade head 2>&1 || {
    echo "WARNING: Migration failed, but continuing to start server..." >&2
    echo "You may need to run migrations manually later" >&2
}

# Start the server (this must succeed)
echo "Starting server on port ${PORT:-8000}..." >&2
exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000} --log-level info
