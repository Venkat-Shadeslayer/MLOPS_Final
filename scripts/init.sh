#!/bin/bash
# ============================================================================
# One-shot project bootstrap
# Run from project root after cloning or fresh directory setup.
# ============================================================================
set -euo pipefail

# Color output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log() { echo -e "${GREEN}[init]${NC} $1"; }
warn() { echo -e "${YELLOW}[warn]${NC} $1"; }
err() { echo -e "${RED}[err]${NC} $1" >&2; }

# Verify we're at project root
if [ ! -f "docker-compose.yml" ]; then
    err "Run this from the project root (where docker-compose.yml lives)."
    exit 1
fi

# 1. .env file
if [ ! -f ".env" ]; then
    log "Creating .env from .env.example"
    cp .env.example .env
else
    warn ".env exists, not overwriting"
fi

# 2. Git init
if [ ! -d ".git" ]; then
    log "Initializing Git repo"
    git init -q
    git add .gitignore .dvcignore .dockerignore
    git commit -q -m "chore: initial gitignore"
else
    warn "Git repo exists"
fi

# 3. Git LFS (for model artifacts too large for regular Git)
if command -v git-lfs &>/dev/null; then
    log "Configuring Git LFS"
    git lfs install
    git lfs track "*.pkl" "*.pt" "*.pth" "*.onnx" "*.parquet"
    git add .gitattributes 2>/dev/null || true
else
    warn "git-lfs not installed — install with 'brew install git-lfs'"
fi

# 4. DVC init
if [ ! -d ".dvc" ]; then
    if command -v dvc &>/dev/null; then
        log "Initializing DVC"
        dvc init -q
        git add .dvc .dvcignore
        git commit -q -m "chore: init DVC" || true
    else
        warn "dvc not installed — install with 'pip install dvc' or 'brew install dvc'"
    fi
else
    warn "DVC already initialized"
fi

# 5. Make shell scripts executable
log "Setting script permissions"
chmod +x scripts/*.sh 2>/dev/null || true

# 6. Verify Docker
if ! command -v docker &>/dev/null; then
    err "Docker not found. Install Docker Desktop."
    exit 1
fi
if ! docker info &>/dev/null; then
    err "Docker daemon not running. Start Docker Desktop."
    exit 1
fi

log "Bootstrap complete."
log "Next: 'docker compose up -d --build' to start the stack."