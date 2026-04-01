#!/bin/bash
set -e

export QWEN35_EXPERIMENT_TAG="${QWEN35_EXPERIMENT_TAG:-}"

PROJECT_DIR="/home/lyl/mathRL"
cd "$PROJECT_DIR"

echo "Experiment tag: ${QWEN35_EXPERIMENT_TAG:-default}"
python src_qwen35/data_prepare.py "$@"
