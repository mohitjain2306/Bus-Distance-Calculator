#!/bin/bash

# deploy.sh - builds and runs the Bus Distance Calculator locally
# usage: bash scripts/deploy.sh

set -e  # stop the script if any command fails

echo "----------------------------------------"
echo "  Bus Distance Calculator - Deploy Script"
echo "----------------------------------------"

cd "$(dirname "$0")/.."

echo ""
echo "[1/3] Stopping any running containers..."
docker compose down --remove-orphans

echo ""
echo "[2/3] Building the Docker image..."
docker compose build --no-cache

echo ""
echo "[3/3] Starting all services..."
docker compose up -d

echo ""
echo "----------------------------------------"
echo "  App is running!"
echo "  Open: http://localhost:80"
echo "  Or direct: http://localhost:8501"
echo "----------------------------------------"
