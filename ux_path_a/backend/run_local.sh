#!/bin/bash
# Bash script to run backend locally with correct PYTHONPATH
# This ensures absolute imports work correctly

# Get project root (parent of ux_path_a/backend)
PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

echo "Starting backend server..."
echo "Project root: $PROJECT_ROOT"
echo "PYTHONPATH: $PYTHONPATH"
echo ""

cd "$(dirname "$0")"
uvicorn main:app --reload --host 0.0.0.0 --port 8000
