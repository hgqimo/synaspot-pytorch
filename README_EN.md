# SYNASPOT

A PyTorch reproduction and improvement of SYNASPOT — a **lightweight, streaming, multi-modal** open-vocabulary keyword spotting (KWS) framework that achieves zero-shot keyword detection via audio-text synergy.

> Corresponding paper: **SYNASPOT: A Lightweight, Streaming Multi-Modal Framework for Keyword Spotting with Audio-Text Synergy**
> Paper: [arXiv:2512.15124](https://arxiv.org/abs/2512.15124)

Core idea: frame-level phoneme labels serve as the "common anchor" between audio and text. The audio encoder outputs `e_a` and the text encoder outputs `e_t`; the two are aligned in a **shared phoneme space**, enabling three enrollment/query modalities:

| Modality | Enrollment | Query |
|---|---|---|
| **T** | Keyword pinyin text | Speech |
| **A** | Keyword enrollment audio | Speech |
| **TA** | Both (fused) | Speech |

---

## 1. Two Available Configurations

This repository provides two **trained, verified, and directly reproducible** configurations covering different deployment constraints:

| | **Option A: SSL / HuBERT** | **Option B: DFSMN student** |
|---|---|---|
| Audio backbone | HuBERT 768-dim @50Hz (frozen, adapter only) | 80-dim online fbank @100→50Hz (trained from scratch) |
| Training | Four-stage pipeline | Four-stage pipeline + SSL teacher distillation |
| **Streaming inference** | ✗ Non-causal | ✓ Strictly causal, streamable |
| **Model size** | 0.84M adapter | 1.46M |
| **T-mode EER** | **4.91%** | 48.34% (≈random, unusable) |
| **A-mode EER** | 5.65% | **4.79%** |
| **TA-mode EER** | **3.91%** | 6.83% |
| Use case | Offline / non-real-time, pure text enrollment | Online wake-up, one enrollment audio |
| Reproduction script | [`scripts/reproduce_A_ssl.sh`](scripts/reproduce_A_ssl.sh) | [`scripts/reproduce_B_dfsmn.sh`](scripts/reproduce_B_dfsmn.sh) |

> EER is reported on the WenetPhrase benchmark **WP_E (easy negatives) / WP_H (hard negatives)** subset with 300 keywords; lower is better.

### Chinese WenetPhrase results: comprehensively exceeding the original paper

On the Chinese WenetPhrase benchmark, our reproduction and improvements **outperform the SYNASPOT reported in the original paper across the board**; Option A's tri-modal fusion (TA) even beats the strongest comparison baseline MM-KWS (4.24%) reported in the paper.

| Model | Enrollment | Params | WP_E EER | WP_H EER |
|---|---|---|---|---|
| Original paper · SYNASPOT (fbank backbone) | A | 0.9M | 18.46% | 36.68% |
| Original paper · SYNASPOT | T | 0.9M | 19.76% | 39.95% |
| Original paper · SYNASPOT | TA | 0.9M | 14.56% | 34.50% |
| Original paper · MM-KWS (comparison method) | T | 3.9M | 4.24% | 26.23% |
| **Ours · Option A (SSL / HuBERT)** | T / A / TA | 0.84M | 4.91% / 5.65% / **3.91%** | 26.99% / 22.94% / 23.33% |
| **Ours · Option B (DFSMN, streaming)** | A | 1.46M | **4.79%** | 26.16% |

Key takeaways (lower EER is better):

- **TA mode**: Option A achieves **3.91%**, a **73% relative reduction** over the paper's SYNASPOT at 14.56%, and even beats the strongest baseline MM-KWS (4.24%).
- **A mode (audio enrollment)**: drops from 18.46% (fbank backbone) in the paper to **5.65%** (SSL) / **4.79%** (DFSMN streaming), a 69% / 74% relative reduction.
- **T mode (pure text enrollment)**: Option A reaches **4.91%**, down sharply from the paper's 19.76%.
- **Fewer parameters**: only 0.84M (Option A) / 1.46M (Option B) trainable parameters.

### Chinese-domain adaptation: changes relative to the original paper

The original paper primarily trains and evaluates on English (LibriPhrase / ARPAbet phones). Adapting it to Chinese (AISHELL-1 training + WenetPhrase evaluation) required the following changes:

1. **Phone inventory switched to pinyin, with tones dropped by default.** English ARPAbet has ~40–71 classes; toned Chinese syllables reach ~1160 classes, which makes the vocabulary sparse, hard to converge, and contributes little to KWS discrimination. We therefore **drop tones** by default (`tone=False`, reducing 1160→420 classes) and can further collapse to **initials only (~24 classes)** to match the ARPAbet scale. Three granularities are exposed for ablation: `initial` / `syllable` / `syllable_tone`.
2. **Direct joint training fails to converge → split into a four-stage pipeline.** AISHELL-1 has 340 speakers; small models trained from scratch on joint objectives tend to collapse. We use a progressive four-stage pipeline: phoneme pretraining → discriminative fine-tuning (AAM margin ramped from 0 + speaker-adversarial GRL) → text alignment → multi-modal contrastive, with per-stage early stopping carrying the best weights.
3. **Frozen Chinese HuBERT backbone to avoid from-scratch collapse.** The paper trains DFSMN+fbank from scratch; under Chinese multi-speaker data this collapses easily, so Option A uses a pretrained Chinese HuBERT (768-dim @50Hz) as a frozen backbone, training only a 0.84M adapter.
4. **Streaming Option B uses SSL-teacher distillation.** To keep streaming, a DFSMN student (80-dim online fbank) is distilled from the HuBERT teacher via embedding-level (frame-wise cosine) + Logit-level KD, transferring phoneme knowledge into the causal encoder.
5. **Frame-level labels require extra alignment.** There are no ready-made Chinese frame-level phoneme alignments, so FunASR (paraformer) recognition + timestamps are used, then character-level pinyin is expanded into 10ms frame labels (uncovered frames marked `<SIL>`).
6. **Decoding windows scaled by frame rate.** HuBERT outputs 50Hz (the paper's fbank is 100Hz), so the scoring windows `w_smooth/w_scoring` scale proportionally (10→5, 50→25) to keep the physical duration unchanged.


**Choosing a configuration**

- Need **pure text enrollment (T mode)** → Option A (Option B's T mode is unusable).
- Need **streaming online inference** → Option B (Option A's HuBERT is bidirectional / non-causal).
- Need **streaming + pure text enrollment** → currently unsolved; see [Known Limitations](#7-known-limitations).

---

## 2. Installation

```bash
conda create -n synaspot python=3.10 -y
conda activate synaspot
pip install -r requirements.txt
```

Verified versions: `torch 2.13.0+cu126`, `torchaudio 2.11.0`, `numpy 2.3.5`, `scikit-learn 1.9.0`, `soundfile 0.14.0`.

---

## 3. Data Preparation

### 3.1 AISHELL-1 (training data)

Download [AISHELL-1](https://www.openslr.org/33/) and place it as:

```
dataset/AISHELL-1/raw/33/data_aishell/
├── wav/*.tar.gz
└── transcript/aishell_transcript_v0.8.txt
```

Then build everything in one shot (extract audio → build manifest → FunASR frame-level phoneme alignment → generate training dataset):

```bash
bash data_prep/run_all.sh
```

This produces `dataset/AISHELL-1/processed/synaspot/{train,dev,test}.jsonl` and `dataset_info.json`.

### 3.2 WenetPhrase (evaluation benchmark)

Evaluation requires the WenetPhrase dataset placed under `dataset/Wenetphrase/`; hard negatives require `wenetphrase_hardneg.json.zip`, specified via the `--hardneg-zip` argument.

### 3.3 Chinese HuBERT (Option A only)

Option A requires a pretrained Chinese HuBERT; download it to `synaspot/Model_File/chinese-hubert-base/`
(you can point to another location with `extract_ssl.py --model-dir`).

> Model weights and datasets are large and excluded from the repository via `.gitignore`.

---

## 4. Reproduction

### Option A: SSL / HuBERT (all three modalities usable, best quality)

```bash
bash scripts/reproduce_A_ssl.sh
# specify GPU:  GPU=3 bash scripts/reproduce_A_ssl.sh
```

Pipeline: offline HuBERT feature extraction → four-stage pipeline training → WenetPhrase evaluation.
Produces `checkpoints/ssl/mixed/best.pt`.

### Option B: DFSMN student + SSL teacher distillation (streaming, best A-mode)

> **Depends on Option A**: needs the teacher checkpoint `checkpoints/ssl/audio/best.pt` and teacher offline features `hubert_feats/` produced by Option A.
> Please **run Option A first**, then run this script.

```bash
bash scripts/reproduce_B_dfsmn.sh
# specify GPUs:  GPUS=2,3 bash scripts/reproduce_B_dfsmn.sh
```

Produces `checkpoints/dfsmn_pipe/mixed/best.pt`.

### Evaluate any checkpoint independently

```bash
python -m synaspot.eval_wenetphrase \
    --ckpt <path/to/mixed/best.pt> --modes T,A,TA
```

The decoder **automatically scales the scoring windows by frame rate** (50Hz → `w_smooth=5, w_scoring=25`), with default fusion weights `1:1:1` — no manual arguments needed.

### Four-stage pipeline

`--pipeline` runs stages sequentially, each with per-stage early stopping that carries the best weights forward:

1. **Phoneme pretraining** — audio-only, `margin=0`, speaker adversarial off
2. **Discriminative fine-tuning** — AAM margin + speaker adversarial (GRL)
3. **Text alignment** — freeze the audio encoder, align `e_t` to the student's own `e_a` via cosine
4. **Multi-modal contrastive** — joint optimization of three modalities (InfoNCE + phoneme classification)

---

## 5. Project Structure

```
synaspot-release/
├── synaspot/
│   ├── config.py          # hyperparameters (backbone, frame rate, loss weights…)
│   ├── model.py           # audio / text / mixed three-branch model
│   ├── dataset.py         # dataset (online fbank / offline cached features, distillation alignment)
│   ├── losses.py          # phoneme AAM, contrastive, alignment, KD losses
│   ├── train.py           # four-stage pipeline training entry
│   ├── decode.py          # streaming decoding and enrollment scoring
│   ├── eval_wenetphrase.py# WenetPhrase open-vocabulary evaluation
│   ├── extract_ssl.py     # offline HuBERT feature extraction
│   ├── distill_online.py  # offline→online distillation (standalone script)
│   ├── tune_windows.py    # scoring-window / fusion-weight grid search
│   ├── diag_mixed.py      # mixed embedding diagnostics
│   └── check_pipeline.py  # pipeline self-check
├── data_prep/             # AISHELL-1 data build pipeline
├── scripts/               # one-shot reproduction scripts for Option A / B
├── static/index.html      # web demo (keyword enrollment + mic barge-in)
├── kws_engine.py          # keyword enrollment storage + streaming detection engine (reusable core)
├── kws_server.py          # REST offline enrollment + WebSocket online streaming detection service
├── dataset/               # training data (AISHELL-1 / Wenetphrase, not checked in)
├── checkpoints/           # training checkpoints (per-option subdirs, not checked in: ssl/ dfsmn_pipe/ …)
└── docs/                  # paper and tuning report
```

---

## 6. Keyword Barge-in Interface (KWS Service)

Targets the **barge-in** scenario of a full-duplex spoken dialogue system: instead of keeping an always-on ASR + hotword recognizer running in the background, lightweight keyword vectors are matched in a streaming fashion and a hit triggers an interruption.

> **Text enrollment relies on the HuBERT semantic space and is only supported by Option A (SSL backbone)**; Option B (DFSMN) supports audio enrollment only — see Section 7, limitations 1 / 2.

### 6.1 Components

| File | Purpose |
| --- | --- |
| `kws_engine.py` | Reusable core: `KeywordStore` (enrollment vectors persisted to disk), `KWSEngine` (enroll / open session), `DetectionSession` (sliding-window scoring + threshold trigger) |
| `kws_server.py` | FastAPI service: REST offline enrollment + WebSocket online streaming detection + static page hosting |
| `static/index.html` | Web demo: text / audio enrollment, mic listening, real-time score and barge-in alert |

### 6.2 Launch

```bash
python kws_server.py \
    --ckpt /path/to/checkpoints/ssl/mixed/best.pt \
    --ssl-model-dir /path/to/chinese-hubert-base \
    --threshold 0.5 \
    --port 8765
```

Open `http://<host>:8765` in a browser to use the demo (enroll → listen → barge-in).

### 6.3 API

**Offline enrollment (REST)**

| Method | Path | Description |
| --- | --- | --- |
| `POST` | `/api/keywords` | Enroll: Form `name` + `text` (text enrollment) or an `audio` file (audio enrollment); optional `threshold` to override the default |
| `GET` | `/api/keywords` | List enrolled keywords |
| `DELETE` | `/api/keywords/{id}` | Delete a keyword |

Enrollment vectors are written to `--store-dir` (default `keyword_store/`, gitignored):
`index.json` stores metadata, `<id>.pt` stores the three embeddings `{ea, et, em}` (missing modalities are `None`).

**Online detection (WebSocket `/ws`)**

| Direction | Message | Description |
| --- | --- | --- |
| → | `{"type":"start","keyword":"<id>"}` | Load the keyword vector and start detection |
| → | PCM bytes (int16 @16kHz mono) | Continuous microphone audio |
| → | `{"type":"threshold","value":0.5}` | Dynamically adjust threshold at runtime |
| → | `{"type":"stop"}` | End the session |
| ← | `{"type":"started", ...}` | Session established (returns keyword name and threshold) |
| ← | `{"type":"score","score":0.31}` | Real-time score (throttled to 80ms) for frontend visualization |
| ← | `{"type":"interrupt","name":"打断","score":0.62}` | **Keyword hit → trigger barge-in** |

### 6.4 Integrating into your own full-duplex system

You can reuse the engine directly without starting the HTTP service:

```python
from kws_engine import KWSEngine

engine = KWSEngine(ckpt_path=..., store_dir=..., ssl_model_dir=...)
kw = engine.register("打断", text="停一下")
ses = engine.new_session(kw["id"])

for pcm_block in mic_stream:            # int16 bytes @16kHz mono
    score, triggered = ses.feed_pcm(pcm_block)
    if triggered:
        interrupt()                     # trigger barge-in
```

`DetectionSession` maintains a sliding window (its length adapts to the enrollment audio duration by default, overridable via `window_sec`). On each incoming audio frame it runs a forward pass over that window, takes the peak score and compares it against the threshold — a hit triggers. The scoring window `w_scoring` likewise adapts to the enrollment template frame count to avoid length-mismatch suppression. `cooldown_sec` (default 2s) suppresses repeated triggers.

---

## 7. Known Limitations

1. **Option B (DFSMN) does not support T mode.** Two root causes (see [`docs/scoring_tuning_report.md`](docs/scoring_tuning_report.md), Section 9.3):
   - Stage-4 joint fine-tuning drifts `e_a`, washing away the text alignment established in stage 3 (27% → 48%);
   - More fundamentally, fbank-driven `e_a` lives in an **acoustic discriminative space** where pinyin text `e_t` cannot stably land; HuBERT's `e_a` is a semantic space to which text aligns naturally.

2. **Streaming + pure text enrollment is currently unsolved.** A causal streaming Conformer backbone (WeNet U2++, 512-dim @25Hz) was tried as a compromise but underperformed (stage-4 dev `acc_ph` only 0.5423, noticeably lower than DFSMN's 0.7204 and SSL's 0.9664); **the related code has been removed from this repository**.

3. **The mixed embedding M is currently a neutral term.** `e_m` is essentially a high-quality reconstruction of `e_a` (cos ≈ 0.95); tri-modal fusion `1:1:1` is on par with dropping M (`1:1:0`). The fix process for the related normalization bug is described in Section 7 of the report.

---

## 8. Citation

Original paper:

> Kewei Li, Yinan Zhong, Xiaotao Liang, Tianchi Dai, Shaofei Xue. *SYNASPOT: A Lightweight, Streaming Multi-Modal Framework for Keyword Spotting with Audio-Text Synergy.* arXiv:2512.15124.

Paper: <https://arxiv.org/abs/2512.15124>. If this repository helps your research, please cite the original paper.