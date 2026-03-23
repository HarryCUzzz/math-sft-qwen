"""
下载评估数据集到本地
使用镜像源避免网络问题
"""

import os
import sys
from pathlib import Path

# 设置 HuggingFace 镜像环境变量（必须在 import datasets 之前）
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"  # 禁用 hf_transfer 避免连接原站

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from datasets import load_dataset

# 数据集保存路径
SAVE_DIR = PROJECT_ROOT / "data" / "eval_datasets"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# 评估数据集配置
EVAL_DATASETS = {
    "math500": {
        "name": "HuggingFaceH4/MATH-500",
        "split": "test",
    },
    "gsm8k": {
        "name": "openai/gsm8k",
        "subset": "main",
        "split": "test",
    },
    "theoremqa": {
        "name": "TIGER-Lab/TheoremQA",
        "split": "test",
    },
}


def download_dataset(dataset_key):
    """下载单个数据集"""
    config = EVAL_DATASETS[dataset_key]
    save_path = SAVE_DIR / dataset_key

    if save_path.exists():
        print(f"✓ {dataset_key} 已存在，跳过下载")
        return True

    print(f"\n{'='*60}")
    print(f"下载数据集: {dataset_key}")
    print(f"{'='*60}")

    try:
        # 加载数据集
        if "subset" in config:
            dataset = load_dataset(
                config["name"],
                config["subset"],
                split=config["split"],
                trust_remote_code=True,
            )
        else:
            dataset = load_dataset(
                config["name"],
                split=config["split"],
                trust_remote_code=True,
            )

        # 保存到本地
        dataset.save_to_disk(str(save_path))
        print(f"✓ {dataset_key} 下载完成，保存至: {save_path}")
        print(f"  样本数: {len(dataset)}")
        return True

    except Exception as e:
        print(f"✗ {dataset_key} 下载失败: {e}")
        return False


def main():
    print("=" * 60)
    print("开始下载评估数据集")
    print("=" * 60)
    print(f"保存路径: {SAVE_DIR}")
    print(f"镜像源: {os.environ.get('HF_ENDPOINT')}")

    success_count = 0
    for dataset_key in EVAL_DATASETS:
        if download_dataset(dataset_key):
            success_count += 1

    print("\n" + "=" * 60)
    print(f"下载完成: {success_count}/{len(EVAL_DATASETS)} 个数据集成功")
    print("=" * 60)


if __name__ == "__main__":
    main()
