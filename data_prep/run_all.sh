#!/usr/bin/env bash
# AISHELL-1 -> SYNASPOT 训练数据 一键流水线
# 用法:
#   bash run_all.sh            # 全量执行
#   bash run_all.sh --smoke    # 冒烟测试(仅标注前 20 条)
set -euo pipefail
cd "$(dirname "$0")"

# 如使用虚拟环境请先激活, 例: source /path/to/venv/bin/activate

NUM_WORKERS=${NUM_WORKERS:-4}   # 进程数
NUM_GPUS=${NUM_GPUS:-4}         # GPU 数
BATCH_SIZE=${BATCH_SIZE:-32}

echo ">>> [1/4] 解压音频"
python 01_extract_audio.py

echo ">>> [2/4] 构建元数据清单"
python 02_build_manifest.py

echo ">>> [3/4] FunASR 标注"
if [[ "${1:-}" == "--smoke" ]]; then
    python 03_funasr_align.py --num-workers 1 --num-gpus 1 --limit 20
else
    python 03_funasr_align.py --num-workers "$NUM_WORKERS" --num-gpus "$NUM_GPUS" \
                              --batch-size "$BATCH_SIZE"
fi

echo ">>> [4/4] 构建训练数据集"
python 04_build_dataset.py

echo ">>> 全部完成"
