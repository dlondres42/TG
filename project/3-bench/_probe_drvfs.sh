#!/usr/bin/env bash
# Probe whether vegeta can write to drvfs vs ext4. Diagnostic only.
set +e
cd /tmp || exit 1
echo "POST http://127.0.0.1:8888/x" > /tmp/t.txt

echo "=== vegeta -> /tmp (ext4) ==="
vegeta attack -targets=/tmp/t.txt -duration=1s -rate=10 -timeout=1s -output=/tmp/r.bin
ls -la /tmp/r.bin
rm -f /tmp/r.bin

OUT=/mnt/c/Users/David/Documents/learning_repos/TG/project/3-bench/results/_drvfs_probe
mkdir -p "$OUT"
echo "=== dir created: ==="
ls -la "$OUT"
echo "=== vegeta -> drvfs ==="
vegeta attack -targets=/tmp/t.txt -duration=1s -rate=10 -timeout=1s -output="$OUT/r.bin"
echo "exit=$?"
ls -la "$OUT/"
rm -rf "$OUT"
