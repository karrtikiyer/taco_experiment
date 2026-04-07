#!/bin/bash
set -euo pipefail

MODEL=${1:-"Qwen/Qwen2.5-Coder-14B-Instruct"}
N=${2:-100}
PREFIX=${3:-"run"}
DTYPE=${4:-"auto"}
ATTN=${5:-""}
DATASET=${6:-"taco"}

# Derive model short name: take part after /, lowercase, strip -instruct/-chat/-base
MODEL_SHORT=$(echo "$MODEL" | rev | cut -d'/' -f1 | rev | tr '[:upper:]' '[:lower:]')
MODEL_SHORT=${MODEL_SHORT%-instruct}
MODEL_SHORT=${MODEL_SHORT%-chat}
MODEL_SHORT=${MODEL_SHORT%-base}

METHODS=(top_p temp_only top_p_only pless pless_norm)

echo "============================================"
echo "Experiment - All Decoding Methods"
echo "============================================"
echo "Dataset:    $DATASET"
echo "Model:      $MODEL"
echo "Model dir:  $MODEL_SHORT"
echo "Problems:   $N"
echo "Prefix:     $PREFIX"
echo "Dtype:      $DTYPE"
echo "Attn:       ${ATTN:-default}"
echo "Methods:    ${METHODS[*]}"
echo "Results in: results/$DATASET/$MODEL_SHORT/${PREFIX}_*/"
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
        --dataset "$DATASET" \
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
echo "Results in: results/$DATASET/$MODEL_SHORT/${PREFIX}_*/"
echo "============================================"
