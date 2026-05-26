#!/usr/bin/env bash
# Export paper-aligned 8-bit MLX LLM for the iOS app (replaces HuggingFace 4-bit pack until Hub updates).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/Mobile-O-App"
python3 export.py --only llm --llm-bits 8 --output-dir exported_models_paper "$@"
echo ""
echo "Next: copy exported_models_paper/llm/ to the iPhone app container:"
echo "  Library/Application Support/Models/llm/"
echo "Or merge into your Xcode install's Application Support/Models after a device build."
