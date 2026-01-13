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

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app:/app/ux_path_a/backend

# Make start script executable (already copied by COPY . /app/)
RUN chmod +x /app/ux_path_a/backend/start.sh

# Expose port (Railway will set PORT env var)
EXPOSE 8000

# Run migrations and start server (from backend directory)
CMD ["./start.sh"]
