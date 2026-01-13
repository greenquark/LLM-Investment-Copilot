#!/bin/bash
set -e

echo "=== Starting container ==="
echo "Working directory: $(pwd)"
echo "Python path: $PYTHONPATH"
echo "PORT: ${PORT:-8000}"

# Run database migrations (don't fail if migrations error)
echo "Running database migrations..."
python -m alembic upgrade head || {
    echo "WARNING: Migration failed, but continuing to start server..."
    echo "You may need to run migrations manually later"
}

# Start the server (this must succeed)
echo "Starting server on port ${PORT:-8000}..."
exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}
