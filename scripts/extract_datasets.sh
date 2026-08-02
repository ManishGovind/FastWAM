#!/usr/bin/env bash
# Extract downloaded FastWAM archives under data/ (symlink -> /work/mgovind1/data/fastwam)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA="${ROOT}/data"

echo "[extract] LIBERO..."
cd "${DATA}/libero_mujoco3.3.2"
shopt -s nullglob
for f in *.tar.gz; do
  echo "  extracting $f"
  tar -xzf "$f"
done

echo "[extract] RoboTwin..."
cd "${DATA}/robotwin2.0"
if compgen -G 'robotwin2.0.tar.gz.part-*' > /dev/null; then
  echo "  concatenating parts + extracting"
  cat robotwin2.0.tar.gz.part-* | tar -xzf -
fi

echo "[extract] done."
ls -la "${DATA}/libero_mujoco3.3.2" | head
ls -la "${DATA}/robotwin2.0" | head
