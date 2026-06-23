#!/usr/bin/env bash
# One-shot setup: install uv + vegeta into ~/.local/bin in WSL2 Ubuntu.
# Idempotent.
set -euo pipefail

mkdir -p "$HOME/.local/bin"

# uv
if ! command -v uv >/dev/null && [ ! -x "$HOME/.local/bin/uv" ]; then
    echo "[setup] installing uv"
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi

export PATH="$HOME/.local/bin:$PATH"

# vegeta
if ! command -v vegeta >/dev/null; then
    echo "[setup] installing vegeta -> ~/.local/bin/"
    URL="https://github.com/tsenart/vegeta/releases/download/v12.12.0/vegeta_12.12.0_linux_amd64.tar.gz"
    curl -sSL "$URL" | tar -xz -C "$HOME/.local/bin" vegeta
    chmod +x "$HOME/.local/bin/vegeta"
fi

# Ensure ~/.local/bin is on PATH and uv uses a parallel venv dir in WSL2.
# .venv-wsl keeps Linux wheels separate from the Windows .venv on the same tree.
if ! grep -q 'HOME/.local/bin' "$HOME/.bashrc" 2>/dev/null; then
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.bashrc"
fi
if ! grep -q 'UV_PROJECT_ENVIRONMENT' "$HOME/.bashrc" 2>/dev/null; then
    echo 'export UV_PROJECT_ENVIRONMENT=.venv-wsl' >> "$HOME/.bashrc"
fi
export UV_PROJECT_ENVIRONMENT=.venv-wsl

echo "uv: $(uv --version)"
echo "vegeta: $(vegeta -version | head -1)"
echo "docker: $(docker version --format '{{.Server.Version}}')"
echo "cpus: $(nproc)"
