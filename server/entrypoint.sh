#!/bin/sh
set -e

# Convert LOG_LEVEL to lowercase for uvicorn
LOG_LEVEL_LOWER=$(echo "${LOG_LEVEL:-info}" | tr '[:upper:]' '[:lower:]')

# Start uvicorn with the specified log level
exec uvicorn graph_service.main:app --host 0.0.0.0 --port "${PORT:-8000}" --log-level "$LOG_LEVEL_LOWER"
