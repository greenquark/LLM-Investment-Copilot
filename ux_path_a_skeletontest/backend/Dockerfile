FROM python:3.11-slim

WORKDIR /app/ux_path_a_skeletontest/backend

RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY ux_path_a_skeletontest/backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app/

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/ux_path_a_skeletontest/backend

RUN chmod +x /app/ux_path_a_skeletontest/backend/start.sh

EXPOSE 8000

CMD ["/app/ux_path_a_skeletontest/backend/start.sh"]

