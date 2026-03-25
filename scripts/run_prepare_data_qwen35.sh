#!/bin/bash
set -e

PROJECT_DIR="/home/lyl/mathRL"
cd "$PROJECT_DIR"

python src_qwen35/data_prepare.py "$@"
