#!/bin/bash
set -e

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install   "transformers>=4.44.0"   "datasets>=2.19.0"   "accelerate>=0.30.0"   "peft>=0.11.0"   "trl>=0.9.0"
pip install bitsandbytes tensorboard tqdm scipy pandas numpy matplotlib sentencepiece protobuf safetensors sympy swanlab "lz4>=4.3.3"
