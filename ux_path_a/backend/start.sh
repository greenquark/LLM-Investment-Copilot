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

# Run database migrations (don't fail if migrations error)
echo "Running database migrations..." >&2
python -m alembic upgrade head 2>&1 || {
    echo "WARNING: Migration failed, but continuing to start server..." >&2
    echo "You may need to run migrations manually later" >&2
}

# Start the server (this must succeed)
echo "Starting server on port ${PORT:-8000}..." >&2
exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000} --log-level info
