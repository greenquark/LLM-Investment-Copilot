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
python -c "
import sys
import traceback
try:
    try:
        from core.database import Base, engine
        from core.models import User, ChatSession, ChatMessage, AuditLog, TokenBudget
    except ImportError:
        from ux_path_a.backend.core.database import Base, engine
        from ux_path_a.backend.core.models import User, ChatSession, ChatMessage, AuditLog, TokenBudget
    
    # Create all tables
    print('Creating tables from models...')
    Base.metadata.create_all(bind=engine)
    print('✓ Base tables created/verified')
except Exception as e:
    print(f'✗ Error creating tables: {e}')
    traceback.print_exc()
    sys.exit(1)
" 2>&1 || {
    echo "WARNING: Failed to create base tables, but continuing..." >&2
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
