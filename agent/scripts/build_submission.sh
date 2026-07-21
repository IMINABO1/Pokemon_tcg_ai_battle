#!/usr/bin/env bash
# Build the Kaggle submission tarball from src/.
#
# Packaged (flattened) layout Kaggle expects:
#   main.py            (top level)
#   deck.csv           (top level)
#   cg/                (all 4 compiled binaries, copied FRESH & unmodified)
#   ptcg_agent/        (runtime package; dev-only modules excluded)
#
# Hard-fails (exit != 0) if the deck is illegal or main.py is not at the tar root.
set -euo pipefail

AGENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_DIR="$(cd "$AGENT_DIR/.." && pwd)"
CG_SRC="$REPO_DIR/sample_submission/sample_submission/cg"
STAGING="$AGENT_DIR/dist/staging"
OUT="$AGENT_DIR/dist/submission.tar.gz"
DECK="${1:-$AGENT_DIR/src/ptcg_agent/deck.csv}"

SIZE_CAP_BYTES=$((197 * 1024 * 1024))   # 197.7 MiB Kaggle cap (conservative)

# Dev-only modules NOT shipped.
DEV_ONLY=("logging_utils.py")

echo ">> wiping staging"
rm -rf "$STAGING"
mkdir -p "$STAGING/ptcg_agent" "$STAGING/cg"

echo ">> copying runtime package (excluding dev-only)"
for f in "$AGENT_DIR"/src/ptcg_agent/*.py; do
    base="$(basename "$f")"
    skip=false
    for d in "${DEV_ONLY[@]}"; do [[ "$base" == "$d" ]] && skip=true; done
    $skip && { echo "   skip $base"; continue; }
    cp "$f" "$STAGING/ptcg_agent/$base"
done

echo ">> copying main.py + deck.csv to top level"
cp "$AGENT_DIR/main.py" "$STAGING/main.py"
cp "$DECK" "$STAGING/deck.csv"

echo ">> copying fresh cg/ binaries"
cp "$CG_SRC"/*.py "$STAGING/cg/"
cp "$CG_SRC"/cg.dll "$CG_SRC"/libcg.dylib "$CG_SRC"/libcg.so "$CG_SRC"/libcg-arm64.so "$STAGING/cg/"

echo ">> stripping __pycache__ / .pyc (stale bytecode must not ship)"
find "$STAGING" -name '__pycache__' -type d -prune -exec rm -rf {} +
find "$STAGING" -name '*.pyc' -delete

echo ">> validating deck legality (hard gate)"
python3 -B - "$STAGING/deck.csv" "$STAGING/cg" <<'PY'
import sys, os
staging_cg_parent = os.path.dirname(sys.argv[2])
sys.path.insert(0, staging_cg_parent)          # so `import cg` resolves to staged copy
sys.path.insert(0, os.path.join(os.path.dirname(sys.argv[2]), "ptcg_agent", ".."))
deck = [int(x) for x in open(sys.argv[1]).read().split("\n")[:60]]
from ptcg_agent import legality
v = legality.deck_violations(deck)
if v:
    print("DECK ILLEGAL:", v); sys.exit(1)
print("   deck legal (60 cards)")
PY

echo ">> creating tarball"
rm -f "$OUT"
find "$STAGING" -name '__pycache__' -type d -prune -exec rm -rf {} +
find "$STAGING" -name '*.pyc' -delete
tar -czf "$OUT" -C "$STAGING" .

echo ">> verifying main.py at tar root"
tar -tzf "$OUT" | grep -qx "./main.py" || { echo "ERROR: main.py not at tar root"; exit 1; }

SIZE=$(wc -c < "$OUT")
echo ">> size: $SIZE bytes (cap $SIZE_CAP_BYTES)"
[[ "$SIZE" -le "$SIZE_CAP_BYTES" ]] || { echo "ERROR: over size cap"; exit 1; }

echo "OK -> $OUT"
