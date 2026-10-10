#!/usr/bin/env bash
# =============================================================================
# 方案 A: SSL / HuBERT 音频主干 (非流式)
#
#   特点: 三模态全部可用 (T 文本 / A 音频 / TA 融合), 效果最好
#   指标: T 4.91%   A 5.65%   TA 3.91%   (WenetPhrase WP_E EER)
#   限制: HuBERT 双向非因果, 不支持流式在线推理
#   产出: $SAVE_DIR/mixed/best.pt
#
# 前置条件:
#   1) 已用 data_prep/run_all.sh 构建训练数据集
#   2) 已下载 Chinese HuBERT 到 $HUBERT_DIR
#   3) 评测集 dataset/Wenetphrase + hardneg zip (见 README)
#
# 用法:
#   bash scripts/reproduce_A_ssl.sh
#   GPU=3 bash scripts/reproduce_A_ssl.sh
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

GPU=${GPU:-0}
SAVE_DIR=${SAVE_DIR:-checkpoints/ssl}

HUBERT_DIR=${HUBERT_DIR:-synaspot/Model_File/chinese-hubert-base}

# 1) 离线提取 HuBERT 特征 (768 维 @50Hz)
for split in train dev test; do
    echo "[A] 提取 HuBERT 特征: ${split}"
    CUDA_VISIBLE_DEVICES=$GPU python -m synaspot.extract_ssl \
        --split "$split" --model-dir "$HUBERT_DIR" --device cuda
done

# 2) 四阶段 pipeline 训练
echo "[A] 四阶段 pipeline 训练 (SSL 主干)"
CUDA_VISIBLE_DEVICES=$GPU python -m synaspot.train \
    --audio-backbone ssl \
    --feat-frame-rate 50 \
    --pipeline \
    --stage-epochs 30 \
    --batch-size 64 \
    --save-dir "$SAVE_DIR" \

# 3) WenetPhrase 评测
echo "[A] 评测"
CUDA_VISIBLE_DEVICES=$GPU python -m synaspot.eval_wenetphrase \
    --ckpt "$SAVE_DIR/mixed/best.pt" --modes T,A,TA

echo "[A] 完成, 最终模型: $SAVE_DIR/mixed/best.pt"
