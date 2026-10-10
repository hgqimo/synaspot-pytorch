#!/usr/bin/env bash
# =============================================================================
# 方案 B: DFSMN 学生 (80 维在线 fbank, 流式因果) + SSL 教师蒸馏
#
#   特点: 轻量 (1.46M 参数), 严格因果可流式部署
#   指标: A 4.79% (WP_E EER)  — 优于论文 fbank 基线 18.46% 与 SSL 教师 5.65%
#         T 48.34% ≈ 随机, 不支持纯文本注册
#         TA 6.83%
#   适用: 可录一条注册音频的唤醒场景
#
# 前置条件:
#   1) 已构建训练数据集 (data_prep/run_all.sh)
#   2) 依赖方案 A 的产物:
#        - 教师检查点   exp/audio/best.pt       (SSL 主干, 方案A 阶段2)
#        - 教师离线特征 hubert_feats/{split}/   (方案A 步骤1)
#      若尚未运行方案 A, 请先执行: bash scripts/reproduce_A_ssl.sh
#
# 用法:
#   bash scripts/reproduce_B_dfsmn.sh
#   GPUS=2,3 bash scripts/reproduce_B_dfsmn.sh
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."          # 切到项目根目录

GPUS=${GPUS:-0,1}
NPROC=${NPROC:-2}
TEACHER=${TEACHER:-checkpoints/ssl/audio/best.pt}
SAVE_DIR=${SAVE_DIR:-checkpoints/dfsmn_pipe}
DATA_DIR=${DATA_DIR:-dataset/AISHELL-1/processed/synaspot}

# ---- 前置校验: 教师检查点与教师离线特征必须就绪 -----------------------------
if [[ ! -f "$TEACHER" ]]; then
    echo "错误: 未找到 SSL 教师检查点 $TEACHER"
    echo "请先运行 scripts/reproduce_A_ssl.sh (或至少训到 pipeline 阶段2)"
    exit 1
fi
if [[ ! -d "$DATA_DIR/hubert_feats/train" ]]; then
    echo "错误: 未找到教师离线特征 $DATA_DIR/hubert_feats/train"
    echo "请先运行 scripts/reproduce_A_ssl.sh 的步骤1 (extract_ssl)"
    exit 1
fi

# ---- 四阶段 pipeline 训练 (DFSMN 从零训练 + SSL 教师蒸馏) ------------------
# 蒸馏信号: E_A 逐帧余弦对齐 (w=1.0) + 音素 Logit KD (w=1.0, T=2.0)
# 学生吃在线 80 维 fbank, 教师吃离线 HuBERT, 二者统一重采样到 50Hz 逐帧对齐
echo ">>> [B] 四阶段 pipeline 训练 (DFSMN 学生 + SSL 教师蒸馏)"
CUDA_VISIBLE_DEVICES=$GPUS torchrun --nproc_per_node=$NPROC --master_port=29500 \
    -m synaspot.train \
    --pipeline \
    --audio-backbone dfsmn \
    --distill-teacher-ckpt "$TEACHER" \
    --distill-teacher-backbone ssl \
    --aam-scale 16 --aam-margin 0.2 \
    --stage-epochs 30 \
    --batch-size 64 --enc-lr-mult 3 \
    --num-workers 2 \
    --save-dir "$SAVE_DIR"

# ---- 评测 ------------------------------------------------------------------
echo ">>> [B] 评测"
CUDA_VISIBLE_DEVICES=${GPUS%%,*} python -m synaspot.eval_wenetphrase \
    --ckpt "$SAVE_DIR/mixed/best.pt" --modes T,A,TA

echo ">>> [B] 完成, 最终模型: $SAVE_DIR/mixed/best.pt"
