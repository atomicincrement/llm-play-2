#!/usr/bin/env bash
# Download Qwen2.5-0.5B model weights from Hugging Face.
# Usage: ./download_model.sh [output_dir]
# Default output dir: models/Qwen2.5-0.5B

set -euo pipefail

REPO="Qwen/Qwen2.5-0.5B"
BASE_URL="https://huggingface.co/${REPO}/resolve/main"
OUT_DIR="${1:-models/Qwen2.5-0.5B}"

FILES=(
    config.json
    generation_config.json
    tokenizer.json
    tokenizer_config.json
    vocab.json
    merges.txt
    model.safetensors
)

mkdir -p "${OUT_DIR}"

# Prefer wget; fall back to curl.
if command -v wget &>/dev/null; then
    dl() { wget -c -q --show-progress -O "$2" "$1"; }
elif command -v curl &>/dev/null; then
    dl() { curl -L --progress-bar -C - -o "$2" "$1"; }
else
    echo "Error: neither wget nor curl found." >&2
    exit 1
fi

echo "Downloading ${REPO} into ${OUT_DIR}/"
for f in "${FILES[@]}"; do
    dest="${OUT_DIR}/${f}"
    if [[ -f "${dest}" ]]; then
        echo "  skip  ${f}  (already exists)"
    else
        echo "  fetch ${f}"
        dl "${BASE_URL}/${f}" "${dest}"
    fi
done

echo "Done."
