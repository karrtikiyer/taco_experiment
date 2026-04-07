#!/bin/bash
set -euo pipefail

MODEL=${1:-"Qwen/Qwen2.5-Coder-14B-Instruct"}
N=${2:-100}
PREFIX=${3:-"gpu_run"}
DTYPE=${4:-"auto"}
ATTN=${5:-""}

METHODS=(top_p temp_only top_p_only pless pless_norm)

echo "============================================"
echo "TACO Experiment - All Decoding Methods"
echo "============================================"
echo "Model:    $MODEL"
echo "Problems: $N"
echo "Prefix:   $PREFIX"
echo "Dtype:    $DTYPE"
echo "Attn:     ${ATTN:-default}"
echo "Methods:  ${METHODS[*]}"
echo "============================================"
echo ""

ATTN_FLAG=""
if [ -n "$ATTN" ]; then
    ATTN_FLAG="--attn-implementation $ATTN"
fi

for METHOD in "${METHODS[@]}"; do
    echo ""
    echo ">>> Starting: $METHOD"
    echo ">>> $(date)"
    echo ""

    PYTHONPATH=src uv run python -m taco_experiment.pipeline \
        --model "$MODEL" \
        --decoding-method "$METHOD" \
        --n-problems "$N" \
        --run-name "${PREFIX}_${METHOD}" \
        --dtype "$DTYPE" \
        $ATTN_FLAG

    echo ""
    echo ">>> Completed: $METHOD"
    echo ">>> $(date)"
    echo ""
done

echo "============================================"
echo "All methods complete!"
echo "Results in: results/${PREFIX}_*/"
echo "============================================"
