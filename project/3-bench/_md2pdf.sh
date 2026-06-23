#!/usr/bin/env bash
# Install pandoc + tectonic to ~/.local/bin (no sudo) and convert the
# Phase 3 report to PDF. Idempotent.
set -euo pipefail

mkdir -p "$HOME/.local/bin"
export PATH="$HOME/.local/bin:$PATH"

if ! command -v pandoc >/dev/null; then
    echo "[setup] installing pandoc -> ~/.local/bin/"
    PANDOC_URL="https://github.com/jgm/pandoc/releases/download/3.5/pandoc-3.5-linux-amd64.tar.gz"
    TMP=$(mktemp -d)
    curl -sSL "$PANDOC_URL" | tar -xz -C "$TMP"
    cp "$TMP"/pandoc-*/bin/pandoc "$HOME/.local/bin/pandoc"
    chmod +x "$HOME/.local/bin/pandoc"
    rm -rf "$TMP"
fi

if ! command -v tectonic >/dev/null; then
    echo "[setup] installing tectonic -> ~/.local/bin/"
    TECTONIC_URL="https://github.com/tectonic-typesetting/tectonic/releases/download/tectonic%400.15.0/tectonic-0.15.0-x86_64-unknown-linux-musl.tar.gz"
    curl -sSL "$TECTONIC_URL" | tar -xz -C "$HOME/.local/bin" tectonic
    chmod +x "$HOME/.local/bin/tectonic"
fi

echo "pandoc:   $(pandoc --version | head -1)"
echo "tectonic: $(tectonic --version)"

SRC="$1"
DST="$2"
echo "[md2pdf] $SRC -> $DST"
pandoc "$SRC" \
    --pdf-engine=tectonic \
    -V geometry:margin=2.5cm \
    -V fontsize=11pt \
    -V colorlinks=true \
    -V linkcolor=blue \
    -V urlcolor=blue \
    -V title-meta="Phase 3 — Benchmark Results" \
    -o "$DST"
ls -la "$DST"
