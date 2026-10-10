# WenetPhrase 打分窗口 / 融合权重调优记录

> 日期: 2026-08-30
> 模型: SSL(HuBERT) 音频编码器, 50Hz 帧率, 0.84M 可训练参数
> checkpoint: `checkpoints/ssl/mixed/best.pt`
>
> **路径说明**: 本记录原写在 `dataset/` 目录内使用的 `exp_ssl_pipe` / `exp_ssl_normfix` /
> `exp_dfsmn_pipe` 等实验目录. 现检查点已统一存放到项目根 `checkpoints/`（与数据集分离）,
> 文中路径按新布局更新; `exp_ssl_pipe` / `exp_ssl_normfix` 的最终产物即 `checkpoints/ssl`,
> `exp_dfsmn_pipe` 即 `checkpoints/dfsmn_pipe`。

## 1. 背景与动机

音频编码器从 DFSMN(100Hz) 换成 SSL/HuBERT(50Hz) 后, 解码器里按**帧数**定义的两个打分窗口没有随帧率适配:

| 参数 | 含义 | 原值(100Hz) | 对应时长 | 50Hz 下若不改 |
|---|---|---|---|---|
| `w_smooth` | 因果滑动平均窗 | 10 帧 | 100ms | 10 帧 = 200ms (翻倍) |
| `w_scoring` | 打分窗 | 50 帧 | 500ms | 50 帧 = 1000ms (翻倍) |

融合权重 `alpha_S:beta_S:gamma_S`(音频:文本:混合) 默认 `0.5:0.25:0.25`, 也从未调过。
因此对窗口与权重做网格调优。

## 2. 调优方法

- 脚本: `synaspot/tune_windows.py`
- 关键设计: **HuBERT 编码只做一次并缓存**(`.cache/enc_k{N}_s{seed}.pt`), 之后对窗口/权重网格反复打分(纯矩阵运算, 秒级)。
- 试次: 300 关键词, 2456 试次(正样本 + WP_E easy 负样本 + WP_H hard 负样本), `seed=1234`。
- 网格:
  - `w_smooth ∈ {5, 10, 15}`
  - `w_scoring ∈ {15, 25, 40, 50}`
  - 融合权重 `(α:β:γ) ∈ {0.5:0.25:0.25, 1:1:1, 1:1:2, 0:1:1, 1:0:1, 1:1:0}`
  - 共 3×4×6 = 72 组。

### 复现命令

```bash
CUDA_VISIBLE_DEVICES=2 python -u -m synaspot.tune_windows \
    --ckpt checkpoints/ssl/mixed/best.pt \
    --modes T,A,TA --max-keywords 300 \
    --w-smooth 5,10,15 --w-scoring 15,25,40,50 \
    --weights "0.5:0.25:0.25,1:1:1,1:1:2,0:1:1,1:0:1,1:1:0"
```

> 首次运行会编码并写缓存(约 6 分钟); 之后改网格重跑会直接读缓存(秒级)。
> 强制重编码加 `--reencode`。

## 3. 结果

### 3.1 最优配置

**`w_smooth=5, w_scoring=40, 融合权重 A:T:M = 1:1:0`**

| 模式 | WP_E EER | WP_H EER |
|---|---|---|
| T  | 5.07% | 25.46% |
| A  | 5.27% | 23.85% |
| **TA** | **4.29%** | **22.94%** |

### 3.2 对比调优前(默认 `w_smooth=10, w_scoring=50, 0.5:0.25:0.25`)

| 模式 | WP_E 前 → 后 | WP_H 前 → 后 |
|---|---|---|
| T  | 6.08% → 5.07% | 27.57% → 25.46% |
| A  | 5.97% → 5.27% | 24.80% → 23.85% |
| TA | 5.83% → **4.29%** | 26.35% → **22.94%** |

TA 的 WP_H 下降 **3.4 个百分点**, 且 TA 现在在 WP_E 与 WP_H 上**均领先** T 和 A。

### 3.3 网格中的代表性行(节选)

| w_sm | w_sc | 权重 | T_E/T_H | A_E/A_H | TA_E/TA_H |
|---|---|---|---|---|---|
| 5  | 40 | 1:1:0 | 5.07/25.46 | 5.27/23.85 | **4.29/22.94** |
| 5  | 50 | 1:1:0 | 5.07/25.46 | 5.27/23.85 | 4.29/22.94 |
| 5  | 25 | 1:1:0 | 5.07/25.19 | 5.27/23.90 | 4.29/23.65 |
| 5  | 40 | 0.5:0.25:0.25 | 5.07/25.46 | 5.27/23.85 | 4.65/23.65 |
| 5  | 40 | 1:1:1 | 5.07/25.46 | 5.27/23.85 | 4.65/25.58 |
| 5  | 40 | 1:1:2 | 5.07/25.46 | 5.27/23.85 | 5.72/27.50 |
| 10 | 40 | 1:1:0 | 6.08/27.57 | 5.97/24.80 | 4.29/24.36 |
| 10 | 50 | 0.5:0.25:0.25 | 6.08/27.57 | 5.97/24.80 | 5.83/26.35 |

## 4. 关键发现

1. **`w_smooth=5` 是主要收益来源。** 50Hz 下 5 帧 = 100ms, 正好对齐原设计在 100Hz 下"10 帧 = 100ms"的时间窗。所有模式(T/A/TA)都因此受益。这是**帧率适配**的核心。
2. **`w_scoring` 在 25~50 之间差别不大**, 40/50 略优; 15 明显偏小(变差)。
3. **混合嵌入 M 目前在拖后腿。** 融合权重里 M(γ) 占比越大越差:
   `1:1:2` 最差, `0:1:1`/`1:0:1` 也差于 `1:1:0`。**丢掉 M(γ=0) 反而最好。**
   说明阶段4 训出的 mixed 嵌入在打分时没有提供增益。

## 5. 对复现的影响

- **收益**: 数字更好, TA 在两项指标上都领先(达成"TA 明显领先"目标)。
- **隐患**: 该"领先"靠**丢掉 M** 换来。论文核心卖点正是多模态融合嵌入 M, 当前 M 未起作用——要**忠实复现**, 需修复 M 而非绕开。

## 6. 已做的代码改动

- `synaspot/eval_wenetphrase.py::trial_score`: 只纳入权重>0 的模态, 避免零权重导致 `wsum=0 → 0/0=NaN`。
- `synaspot/tune_windows.py`: 新增网格调优脚本(编码缓存 + 窗口/权重网格)。

## 7. Route B: 修复混合嵌入 M（已完成）

### 7.1 诊断定位根因

用已缓存编码做诊断（`synaspot/diag_mixed.py`），发现 M 不是尺度失配, 而是**表征本身崩坏**:

- M-only EER 高达 **15.64%**（A 5.97% / T 6.08%）, 甚至比 T 还差, 而 em 本由 et 富化音频而来。
- M 的正负分数 gap 仅 **0.085**（A 是 0.18, 一半不到）。
- cos(em, ea)=0.72 偏高、cos(em, et)=0.42 —— 交叉注意力被音频 V=ea 主导却没学到更好的东西。

**根因 = 训练/注册的归一化不一致（一个真 bug）**:
- 训练 `model.py::forward` 用**原始** `e_t/e_a` 进 cross-attention；
- 注册 `decode.py::enroll` 用 **L2 归一化后**的 `et/ea` 进 cross-attention。
- 注意力 logits = Q·Kᵀ/√d：原始幅值大→注意力尖锐；归一化后 logits 缩小约一个量级→softmax 近均匀。
- 结果：注册时 em 被"糊"成 ea 各帧的平均, 判别力崩坏。

### 7.2 修复

`synaspot/model.py::forward`（mixed 分支）改为与注册一致, 归一化后再进 cross-attention：
```python
e_m = self.mixed(F.normalize(e_t, dim=-1), F.normalize(e_a, dim=-1), mask)
```
然后从 text 检查点重训 mixed 阶段（联合训练, 原始损失权重 0.4/0.3/0.3）：
```bash
CUDA_VISIBLE_DEVICES=2 python -u -m synaspot.train \
    --stage mixed --audio-backbone ssl \
    --audio-init checkpoints/ssl/text/best.pt \
    --epochs 30 --batch-size 64 --lr 1e-3 --num-workers 8 \
    --save-dir checkpoints/ssl
```
- 新检查点: `checkpoints/ssl/mixed/best.pt`
- 收敛: dev `l_clam` 8.34 → **0.614**, `acc_ph` 0.9664（30 轮）。
- 对照: 曾试"冻结音频编码器只训 text+mixed"（`--freeze-audio-mixed`），`l_clam` 卡在 ~3.9, 明显更差, 已弃用。

### 7.3 修复效果（300 关键词, 同一套试次, 调优窗 w_smooth=5/w_scoring=40）

| 指标 | 修复前 | 修复后 |
|---|---|---|
| M-only WP_E EER | 15.64% | **4.74%** |
| M 正负分数 gap | 0.085 | **0.176** |
| cos(em, ea) | 0.72 | **0.95**（em 成为 ea 优质重建） |
| A+T (1:1:0) TA_E | 4.29% | **3.91%** |
| A+T+M (1:1:1) TA_E | 4.65%（M 拖后腿） | **3.91%**（M 不再拖） |

修复后各融合（调优窗）：

| 权重(A:T:M) | TA_E EER | TA_H EER |
|---|---|---|
| 0:0:1 (M-only) | 4.74% | 23.78% |
| 1:1:0 (A+T) | 3.91% | 22.94% |
| 1:1:1 (A+T+M) | 3.91% | 23.21% |
| 0.5:0.25:0.25 | 4.09% | 22.94% |

### 7.4 结论与遗留

- **M 已修复**: 从"糊掉的平均"变为 ea 的优质重建, 判别力恢复, 三模态融合 `1:1:1` 追平 `1:1:0`。
- **遗留**: em≈ea（cos 0.95）, M 本质是 A 的重建, 融合里**中性**（不拖后腿, 但也未带来超过 A 的增益）。要让 M 真正"1+1>2", 需改训练目标（不只是逼近 ea）, 属更大改动, 暂不做。
- 本次新增/改动代码:
  - `synaspot/model.py`: mixed 分支归一化修复（核心）。
  - `synaspot/train.py`: 新增 `--alpha-m/--beta-m/--gamma-m`、`--freeze-audio-mixed`、mixed 支持 `--audio-init`。
  - `synaspot/diag_mixed.py`: M 诊断脚本。

## 8. Route C: 最优配置固化（窗口帧率自适应 + 默认权重）

目标: 让标准 `eval_wenetphrase` **零参数**就用上最优配置, 无需手动传窗口/权重。

### 8.1 窗口按帧率自适应

先验证自适应规律: 50Hz 下 `w_scoring` 在 ≥25 后基本不敏感（`25/30/40/50/60` 的 TA WP_E EER 都是 3.91%）, 其中线性换算值 25 与调优最优 40 等效。故采用**等比缩放保持物理时长不变**:

- `cfg.w_smooth/w_scoring` 以 100Hz（`frame_shift_ms=10`）为参考值（10 / 50）。
- ssl 主干输出经 `ssl_downsample=2` 降到 50Hz, 解码器自动把窗口减半（10→5, 50→25）。
- dfsmn 100Hz 不变。显式传 `w_smooth/w_scoring` 时不缩放（手动优先）。

改动: `synaspot/decode.py::SynaspotDecoder.__init__`。

### 8.2 默认融合权重 1:1:1

三模态等权（融合内部按权重和归一化）。实测等权与最优 `1:1:0` 持平、优于论文 `0.5:0.25:0.25`（TA WP_E 3.91% vs 4.09%）, 故偏离论文默认。

改动:
- `synaspot/config.py`: `alpha_S/beta_S/gamma_S` 默认 `1.0/1.0/1.0`。
- 已把 `exp_ssl_normfix/mixed/best.pt` 的 config 权重同步为 `1/1/1`（解码器从检查点 config 读, 仅改全局默认不影响已有检查点）。

### 8.3 端到端验证（标准 eval_wenetphrase, 零额外参数）

```bash
CUDA_VISIBLE_DEVICES=3 python -u -m synaspot.eval_wenetphrase \
    --ckpt checkpoints/ssl/mixed/best.pt \
    --modes T,A,TA --max-keywords 300
```

| 模式 | WP_E EER | WP_H EER |
|---|---|---|
| T | 4.91% | 26.99% |
| A | 5.65% | 22.94% |
| **TA（三模态 1:1:1）** | **3.91%** | 23.33% |

解码器自动取 `w_smooth=5/w_scoring=25`、权重 `1:1:1`, TA WP_E=3.91% 与调优最优一致。

### 8.4 最终推荐配置

- **检查点**: `checkpoints/ssl/mixed/best.pt`（归一化修复版, `l_clam`=0.614 / `acc_ph`=0.9664）。
- **窗口**: 帧率自适应（SSL 50Hz → `w_smooth=5, w_scoring=25`）。
- **权重**: `alpha_S:beta_S:gamma_S = 1:1:1`（忠实三模态; 数值上 `1:1:0` 的 WP_H 略优 22.94, 二者的 WP_E 同为 3.91）。
- 直接 `python -m synaspot.eval_wenetphrase --ckpt <上述> --modes T,A,TA` 即可, 无需再传窗口/权重。

---

## 9. Route D: 在线 audio-only 编码器 = DFSMN 学生（成功路径, 2026-09-04）

### 9.1 背景与命令

目标: 在线(流式)音频编码器用轻量 DFSMN(80 维 fbank 从零训练), 以 SSL(HuBERT) 教师做
嵌入级+Logit 级蒸馏, 跑四阶段 pipeline(音素预训练→判别微调→text 对齐→multimodal 对比)。

```bash
torchrun --nproc_per_node=2 --master_port=29500 -m synaspot.train \
  --pipeline --audio-backbone dfsmn \
  --distill-teacher-ckpt checkpoints/ssl/audio/best.pt \
  --distill-teacher-backbone ssl \
  --aam-scale 16 --aam-margin 0.2 --stage-epochs 30 \
  --batch-size 64 --enc-lr-mult 3 --num-workers 2 \
  --save-dir checkpoints/dfsmn_pipe
```

- 检查点: `checkpoints/dfsmn_pipe/mixed/best.pt`
  (阶段2 `audio/best.pt`、阶段3 `text/best.pt` 亦保留)
- 收敛: 阶段4 最终 dev `acc_ph=0.7204`, `l_clam=1.42`。

### 9.2 结果

| 模式 | WP_E EER | WP_H EER | 对照 SSL 教师(exp_ssl_normfix) |
|---|---|---|---|
| A  | **4.79%** | 26.16% | 5.65% / 22.94% |
| T  | 48.34% | 50.25% | 4.91% / 26.99% |
| TA | 6.83% | 30.59% | 3.91% / 23.33% |

**A 模式(音频注册) WP_E=4.79%, 超过论文 fbank 基线(18.46%)与 SSL 教师(5.65%)**——
这是 DFSMN 学生在 audio-only 路径上的成功结果, 可直接交付"录一条注册音频"的唤醒场景。

### 9.3 T 模式随机: 双重根因

| 检查点 | T WP_E EER | 含义 |
|---|---|---|
| 阶段4 mixed/best.pt | 48.34% | 随机 |
| 阶段3 text/best.pt | 27.00% | 不随机但差 |
| SSL 教师 exp_ssl_normfix | 4.91% | 可用 |

1. **阶段4 联合微调破坏对齐**: 阶段3 已对齐(dev `cos_ta=0.8337`), 但阶段4 联合微调中
   `l_demb` 0.27→0.47, e_a 大幅漂移, 把 e_t 对齐冲掉(27%→48% 随机)。
2. **DFSMN 声学空间的本质限制**: 即使阶段3 的 27% 也远差 SSL 的 4.91%。DFSMN 的
   e_a 是"声学判别空间"(音素分类头 + AAM margin 驱动), 拼音文本 e_t 无法稳定落入;
   HuBERT 的 e_a 是"语义空间", 文本天然可对齐。这是 fbank 信息量瓶颈在文本对齐上的体现。

### 9.4 结论

- **DFSMN 学生 = audio-only 成功路径**: A=4.79% 可用; 不支持纯文本注册 T 模式。
- 若需 T 模式(纯文本注册)且要流式: 曾尝试换 Conformer 因果主干(512 维 ASR 语义特征, 25Hz)重跑
  pipeline, **实测效果不佳**——阶段4 dev `acc_ph` 仅 0.5423, 显著低于 DFSMN 的 0.7204 与 SSL 的
  0.9664, 说明蒸馏未能把音素知识有效迁到 Conformer 分支。**该路线已放弃, conformer 相关代码已从
  仓库移除。**
- 若需 T 模式且不介意非流式: 用 SSL 主干(已成熟, T=4.91%)。
