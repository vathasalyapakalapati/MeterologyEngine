#!/bin/sh
cd "$(dirname "$0")" && .venv/bin/uvicorn backend.app:app --port 8000
