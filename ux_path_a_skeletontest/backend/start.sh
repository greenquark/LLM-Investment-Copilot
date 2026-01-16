#!/bin/bash
set -e

export PYTHONUNBUFFERED=1
exec 1>&2

echo "=== Skeleton backend starting ===" >&2
echo "Working directory: $(pwd)" >&2
echo "Python version: $(python --version)" >&2
echo "PORT: ${PORT:-8000}" >&2

exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000} --log-level info

