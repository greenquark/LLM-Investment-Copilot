# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory to backend (but we'll have access to entire repo)
WORKDIR /app/ux_path_a/backend

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY ux_path_a/backend/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy entire repository (so project root's core/ is available)
# This allows access to core/models/, core/data/, core/strategy/, etc.
COPY . /app/

# Verify critical files exist (helps debug import issues)
RUN echo "=== Verifying critical files ===" && \
    ls -la /app/core/strategy/llm_trend_detection.py || echo "WARNING: core/strategy/llm_trend_detection.py not found!" && \
    ls -la /app/core/strategy/registry.py || echo "WARNING: core/strategy/registry.py not found!" && \
    ls -la /app/core/models/bar.py || echo "WARNING: core/models/bar.py not found!" && \
    ls -la /app/core/data/factory.py || echo "WARNING: core/data/factory.py not found!" && \
    echo "=== Verification complete ==="

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app:/app/ux_path_a/backend

# Make start script executable (already copied by COPY . /app/)
# Verify the file exists and make it executable
RUN ls -la /app/ux_path_a/backend/start.sh && chmod +x /app/ux_path_a/backend/start.sh

# Expose port (Railway will set PORT env var)
EXPOSE 8000

# Run migrations and start server (use absolute path to be safe)
CMD ["/app/ux_path_a/backend/start.sh"]
