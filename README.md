# SYNASPOT

SYNASPOT 的 PyTorch 复现与改进：一个**轻量、可流式、多模态**的开放词表关键词 spotting（KWS）框架，通过音频-文本协同实现 zero-shot 关键词检测。

> 对应论文：**SYNASPOT: A Lightweight, Streaming Multi-Modal Framework for Keyword Spotting with Audio-Text Synergy**
> 论文地址：[arXiv:2512.15124](https://arxiv.org/abs/2512.15124)

核心思路：把语音帧级音素标签作为"音频与文本的公共锚点"，音频编码器输出 `e_a`、文本编码器输出 `e_t`，二者在**共享音素空间**对齐，从而支持三种注册/查询模态：

| 模态 | 注册方式 | 查询方式 |
|---|---|---|
| **T** | 关键词拼音文本 | 语音 |
| **A** | 关键词注册音频 | 语音 |
| **TA** | 两者都用（融合） | 语音 |

---

## 一、两种可用配置

本仓库提供两套**已训练验证、可直接复现**的配置，覆盖不同部署约束：

| | **方案 A：SSL / HuBERT** | **方案 B：DFSMN 学生** |
|---|---|---|
| 音频主干 | HuBERT 768 维 @50Hz（冻结，只训适配器） | 80 维在线 fbank @100→50Hz（从零训练） |
| 训练方式 | 四阶段 pipeline | 四阶段 pipeline + SSL 教师蒸馏 |
| **流式推理** | ✗ 非因果 | ✓ 严格因果，可流式 |
| **模型量级** | 适配器 0.84M | 1.46M |
| **T 模式 EER** | **4.91%** | 48.34%（≈随机，不可用） |
| **A 模式 EER** | 5.65% | **4.79%** |
| **TA 模式 EER** | **3.91%** | 6.83% |
| 适用场景 | 离线/非实时、需要纯文本注册 | 在线唤醒、可录一条注册音频 |
| 复现脚本 | [`scripts/reproduce_A_ssl.sh`](scripts/reproduce_A_ssl.sh) | [`scripts/reproduce_B_dfsmn.sh`](scripts/reproduce_B_dfsmn.sh) |

> EER 为 WenetPhrase 评测集 **WP_E（easy 负样本）/ WP_H（hard 负样本）**、300 关键词的结果，越低越好。

### 中文 WenetPhrase 实验结果：全面超过原论文

在中文 WenetPhrase 评测集上，本仓库的复现与改进结果**全面优于原论文报告的 SYNASPOT**；方案 A 的三模态融合（TA）还超过了论文报道的最强对比基线 MM-KWS（4.24%）。

| 模型 | 注册模态 | 参数量 | WP_E EER | WP_H EER |
|---|---|---|---|---|
| 原论文 · SYNASPOT（fbank 主干） | A | 0.9M | 18.46% | 36.68% |
| 原论文 · SYNASPOT | T | 0.9M | 19.76% | 39.95% |
| 原论文 · SYNASPOT | TA | 0.9M | 14.56% | 34.50% |
| 原论文 · MM-KWS（对比方法） | T | 3.9M | 4.24% | 26.23% |
| **本仓库 · 方案 A（SSL / HuBERT）** | T / A / TA | 0.84M | 4.91% / 5.65% / **3.91%** | 26.99% / 22.94% / 23.33% |
| **本仓库 · 方案 B（DFSMN，流式）** | A | 1.46M | **4.79%** | 26.16% |

关键结论（EER 越低越好）：

- **TA 模式**：方案 A 达 **3.91%**，相比原论文 SYNASPOT 的 14.56% **相对降幅 73%**，并超过论文最强基线 MM-KWS（4.24%）。
- **A 模式（音频注册）**：从原论文的 18.46%（fbank 主干）降到 **5.65%**（SSL）/ **4.79%**（DFSMN 流式），相对降幅 69% / 74%。
- **T 模式（纯文本注册）**：方案 A 达 **4.91%**，从原论文的 19.76% 大幅下降。
- **参数更省**：方案 A 仅 0.84M、方案 B 1.46M 可训练参数，均保持轻量。

### 中文领域适配：相对原论文的改动

原论文主要在英文（LibriPhrase / ARPAbet 音素）上训练与评测。直接搬到中文（AISHELL-1 训练 + WenetPhrase 评测）时，做了如下适配：

1. **音素体系换成拼音，且默认去声调。** 英文 ARPAbet 约 40~71 类；中文若保留声调的带调音节达 ~1160 类，词表稀疏、难收敛，且声调对 KWS 判别贡献有限。故默认**去掉声调**（`tone=False`，类别 1160→420），并可进一步折叠为**仅声母 ~24 类**以贴近论文 ARPAbet 规模。提供三档粒度开关消融：`initial` / `syllable` / `syllable_tone`。
2. **直接联合训练难收敛 → 拆四阶段 pipeline。** AISHELL-1 为 340 说话人多说话人数据，小模型从零联合训练易塌缩。改为递进式四阶段：音素预训练 → 判别微调（AAM margin 从 0 平滑递增 + 说话人对抗 GRL）→ 文本对齐 → 多模态对比，逐阶段早停并携带最优权重。
3. **引入 Chinese HuBERT 冻结主干规避从零塌缩。** 原论文用 DFSMN+fbank 从零训练；中文多说话人下从零训 DFSMN 易塌缩，方案 A 改用预训练 Chinese HuBERT（768 维 @50Hz）作冻结主干、只训 0.84M 适配器，语义特征自带判别力。
4. **流式方案 B 用 SSL 教师蒸馏。** 为兼顾流式，用 DFSMN 学生（80 维在线 fbank）对 HuBERT 教师做嵌入级（逐帧余弦）+ Logit 级 KD 蒸馏，把音素知识迁入因果编码器。
5. **帧级标签需额外对齐。** 中文无现成帧级音素对齐，用 FunASR（paraformer）做识别 + 时间戳对齐，再把字级拼音展开为 10ms 帧级标签（未覆盖帧记 `<SIL>`）。
6. **解码窗口按帧率自适应。** HuBERT 输出 50Hz（论文 fbank 为 100Hz），打分窗口 `w_smooth/w_scoring` 等比缩放（10→5、50→25），保持物理时长不变。


**选型建议**

- 需要**纯文本注册（T 模式）** → 方案 A（方案 B 的 T 模式不可用）。
- 需要**流式在线推理** → 方案 B（方案 A 的 HuBERT 双向非因果）。
- 需要**流式 + 纯文本注册** → 目前无解，见[已知限制](#七已知限制)。

---

## 二、环境安装

```bash
conda create -n synaspot python=3.10 -y
conda activate synaspot
pip install -r requirements.txt
```

实测通过版本：`torch 2.13.0+cu126`、`torchaudio 2.11.0`、`numpy 2.3.5`、`scikit-learn 1.9.0`、`soundfile 0.14.0`。

---

## 三、数据准备

### 3.1 AISHELL-1（训练数据）

下载 [AISHELL-1](https://www.openslr.org/33/) 并放置为：

```
dataset/AISHELL-1/raw/33/data_aishell/
├── wav/*.tar.gz
└── transcript/aishell_transcript_v0.8.txt
```

然后一键构建（解压音频 → 构建清单 → FunASR 帧级音素对齐 → 生成训练数据集）：

```bash
bash data_prep/run_all.sh
```

产出 `dataset/AISHELL-1/processed/synaspot/{train,dev,test}.jsonl` 与 `dataset_info.json`。

### 3.2 WenetPhrase（评测集）

评测需要 WenetPhrase 数据集，放置于 `dataset/Wenetphrase/`；难负样本需 `wenetphrase_hardneg.json.zip`，通过 `--hardneg-zip` 指定路径。

### 3.3 Chinese HuBERT（仅方案 A）

方案 A 需要预训练 Chinese HuBERT，下载后放到 `synaspot/Model_File/chinese-hubert-base/`
（可用 `extract_ssl.py --model-dir` 指定其它位置）。

> 模型权重与数据集体积较大，已由 `.gitignore` 排除，不会入库。

---

## 四、复现

### 方案 A：SSL / HuBERT（三模态全可用，效果最优）

```bash
bash scripts/reproduce_A_ssl.sh
# 指定 GPU:  GPU=3 bash scripts/reproduce_A_ssl.sh
```

流程：离线提取 HuBERT 特征 → 四阶段 pipeline 训练 → WenetPhrase 评测。
产出 `checkpoints/ssl/mixed/best.pt`。

### 方案 B：DFSMN 学生 + SSL 教师蒸馏（流式，A 模式最优）

> **依赖方案 A**：需要 A 产出的教师检查点 `checkpoints/ssl/audio/best.pt` 与教师离线特征 `hubert_feats/`。
> 请**先执行方案 A**，再运行本脚本。

```bash
bash scripts/reproduce_B_dfsmn.sh
# 指定 GPU:  GPUS=2,3 bash scripts/reproduce_B_dfsmn.sh
```

产出 `checkpoints/dfsmn_pipe/mixed/best.pt`。

### 单独评测任意检查点

```bash
python -m synaspot.eval_wenetphrase \
    --ckpt <path/to/mixed/best.pt> --modes T,A,TA
```

解码器会**按帧率自动缩放打分窗口**（50Hz → `w_smooth=5, w_scoring=25`），融合权重默认 `1:1:1`，无需手动传参。

### 四阶段 pipeline 说明

`--pipeline` 依次执行，每阶段平台早停并将最优权重带入下一阶段，各自落盘到
`<save-dir>/` 下对应子目录（`best.pt` = 该阶段平台最优，`last.pt` = 最后 epoch，`tb/` = TensorBoard 日志）：

| 阶段 | 子目录 | 说明 |
|---|---|---|
| 1. 音素预训练 | `s1_ph/` | audio-only，`margin=0`，关闭说话人对抗 |
| 2. 判别微调 | `audio/` | 加入 AAM margin + 说话人对抗（GRL）；audio-only 模型（方案 B 的教师来源） |
| 3. 文本对齐 | `text/` | 冻结音频编码器，用余弦把 `e_t` 对齐到学生自己的 `e_a` |
| 4. 多模态对比 | `mixed/` | 三模态联合优化（InfoNCE + 音素分类）；**最终部署模型** |

### checkpoints 目录说明

`checkpoints/` 下按方案分子目录，本地目录由 `.gitignore` 排除、不入库；预训练权重已上传至 HuggingFace：

| 目录 | 对应方案 | 说明 |
|---|---|---|
| `ssl/` | 方案 A（SSL / HuBERT） | 三模态全可用，最终模型 `ssl/mixed/best.pt` |
| `dfsmn_pipe/` | 方案 B（DFSMN 学生 + 蒸馏） | 流式，最终模型 `dfsmn_pipe/mixed/best.pt` |

**预训练权重**：模型仓库 [HuGuanqiang/synaspot](https://huggingface.co/HuGuanqiang/synaspot)（当前私有）。下载到本地 `checkpoints/` 对应位置：

```bash
huggingface-cli login
huggingface-cli download HuGuanqiang/synaspot --local-dir checkpoints
```

---

## 五、项目结构

```
synaspot-release/
├── synaspot/
│   ├── config.py          # 超参配置（主干选择、帧率、损失权重…）
│   ├── model.py           # 音频/文本/混合三分支模型
│   ├── dataset.py         # 数据集（在线 fbank / 离线缓存特征、蒸馏对齐）
│   ├── losses.py          # 音素 AAM、对比学习、对齐、KD 损失
│   ├── train.py           # 四阶段 pipeline 训练入口
│   ├── decode.py          # 流式解码与注册打分
│   ├── eval_wenetphrase.py# WenetPhrase 开放词表评测
│   ├── extract_ssl.py     # 离线提取 HuBERT 特征
│   ├── distill_online.py  # 离线→在线蒸馏（独立脚本）
│   ├── tune_windows.py    # 打分窗口/融合权重网格调优
│   ├── diag_mixed.py      # 混合嵌入诊断
│   └── check_pipeline.py  # 流水线自检
├── data_prep/             # AISHELL-1 数据构建流水线
├── scripts/               # 方案 A / B 一键复现脚本
├── static/index.html      # Web 演示界面（注册关键词 + 麦克风打断）
├── kws_engine.py          # 关键词注册存储 + 流式检测引擎（可复用核心）
├── kws_server.py          # REST 离线注册 + WebSocket 在线流式检测服务
├── dataset/               # 训练数据（AISHELL-1 / Wenetphrase，不入库）
├── checkpoints/           # 训练检查点（按方案分子目录，不入库：ssl/ dfsmn_pipe/ …）
└── docs/                  # 论文与调优报告
```

---

## 六、关键词打断接口（KWS 服务）

面向全双工语音对话系统的 **打断（barge-in）** 场景：无需常开 ASR + 热词在后台持续识别，改用轻量关键词向量做流式匹配，命中关键词即触发打断。

> **文本注册依赖 HuBERT 语义空间，仅方案 A（SSL 主干）支持**；方案 B（DFSMN）只支持音频注册，见第七节限制 1 / 2。

### 6.1 组成

| 文件 | 作用 |
| --- | --- |
| `kws_engine.py` | 可复用核心：`KeywordStore`（注册向量落盘）、`KWSEngine`（注册 / 开会话）、`DetectionSession`（滑窗打分 + 阈值触发） |
| `kws_server.py` | FastAPI 服务：REST 离线注册 + WebSocket 在线流式检测 + 静态页面托管 |
| `static/index.html` | Web 演示：文本 / 音频注册、打开麦克风监听、实时得分与打断提示 |

### 6.2 启动

```bash
python kws_server.py \
    --ckpt /path/to/checkpoints/ssl/mixed/best.pt \
    --ssl-model-dir /path/to/chinese-hubert-base \
    --threshold 0.5 \
    --port 8765
```

浏览器打开 `http://<host>:8765` 即可使用演示界面（注册 → 监听 → 打断）。

### 6.3 接口

**离线注册（REST）**

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `POST` | `/api/keywords` | 注册：Form `name` + `text`（文本注册）或 `audio` 文件（音频注册）；可选 `threshold` 覆盖默认阈值 |
| `GET` | `/api/keywords` | 列出已注册关键词 |
| `DELETE` | `/api/keywords/{id}` | 删除关键词 |

注册向量写入 `--store-dir`（默认 `keyword_store/`，已 gitignore）：
`index.json` 存元信息，`<id>.pt` 存三种嵌入 `{ea, et, em}`（缺失模态为 `None`）。

**在线检测（WebSocket `/ws`）**

| 方向 | 消息 | 说明 |
| --- | --- | --- |
| → | `{"type":"start","keyword":"<id>"}` | 载入该关键词向量，开始检测 |
| → | PCM bytes（int16 @16kHz 单声道） | 持续推送麦克风音频 |
| → | `{"type":"threshold","value":0.5}` | 运行中动态调阈值 |
| → | `{"type":"stop"}` | 结束会话 |
| ← | `{"type":"started", ...}` | 会话建立（返回关键词名与阈值） |
| ← | `{"type":"score","score":0.31}` | 实时得分（节流 80ms），供前端可视化 |
| ← | `{"type":"interrupt","name":"打断","score":0.62}` | **命中关键词 → 触发打断** |

### 6.4 集成到自有全双工系统

不启 HTTP 服务也可直接复用引擎：

```python
from kws_engine import KWSEngine

engine = KWSEngine(ckpt_path=..., store_dir=..., ssl_model_dir=...)
kw = engine.register("打断", text="停一下")
ses = engine.new_session(kw["id"])

for pcm_block in mic_stream:            # int16 bytes @16kHz 单声道
    score, triggered = ses.feed_pcm(pcm_block)
    if triggered:
        interrupt()                     # 触发打断
```

`DetectionSession` 维护滑动窗口（长度默认自适应为注册音频时长，可用 `window_sec` 覆盖），每来一帧音频就对该窗口整体前向一次，取窗口内峰值得分与阈值比较，超出即触发；打分窗口 `w_scoring` 同样自适应为注册模板帧数，避免长度失配压低得分。`cooldown_sec`（默认 2s）抑制重复触发。

---

## 七、已知限制

1. **方案 B（DFSMN）不支持 T 模式。** 根因有二（详见 [`docs/scoring_tuning_report.md`](docs/scoring_tuning_report.md) 第 9.3 节）：
   - 阶段 4 联合微调会让 `e_a` 漂移，把阶段 3 建立的文本对齐冲掉（27% → 48%）；
   - 更本质的是，fbank 驱动的 `e_a` 是**声学判别空间**，拼音文本的 `e_t` 难以稳定落入；而 HuBERT 的 `e_a` 是语义空间，文本天然可对齐。

2. **流式 + 纯文本注册目前无解。** 曾尝试因果流式 Conformer 主干（WeNet U2++，512 维 @25Hz）作为折中方案，但实测效果不佳（阶段 4 dev `acc_ph` 仅 0.5423，显著低于 DFSMN 的 0.7204 与 SSL 的 0.9664），**相关代码已从本仓库移除**。

3. **混合嵌入 M 目前是中性项。** `e_m` 本质是对 `e_a` 的优质重建（cos ≈ 0.95），三模态融合 `1:1:1` 与去掉 M 的 `1:1:0` 基本持平。相关归一化 bug 的修复过程见报告第 7 节。

---

## 八、引用

原论文：

> Kewei Li, Yinan Zhong, Xiaotao Liang, Tianchi Dai, Shaofei Xue. *SYNASPOT: A Lightweight, Streaming Multi-Modal Framework for Keyword Spotting with Audio-Text Synergy.* arXiv:2512.15124.

论文地址：<https://arxiv.org/abs/2512.15124>。若本仓库对你的研究有帮助，请引用原论文。
