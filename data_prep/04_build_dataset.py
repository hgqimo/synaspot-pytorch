"""Step 4: 构建 SYNASPOT 训练数据集.

输入: transcripts.jsonl (官方转写) + alignments.jsonl (FunASR 时间戳)
输出 (写入 processed/synaspot/):
    pinyin_vocab.json   拼音词表 {token: id}, 含 <PAD>/<UNK>/<SIL>
    speaker_map.json    说话人 -> 分类 id (仅 train 说话人, 供梯度反转分类器)
    dataset_info.json   数据集统计与特征配置(帧移/帧长等)
    {train,dev,test}.jsonl  最终清单, 每行:
        {"key", "wav", "split", "speaker", "spk_id", "text"(asr文本),
         "pinyin": [id,...], "label": 帧标签npy路径, "num_frames", "duration"}
    labels/{split}/{key}.npy  int16 帧级拼音标签(10ms/帧, 未覆盖帧为 <SIL>)

说明:
- 帧级标签与训练时 kaldi fbank(frame_length=25ms, frame_shift=10ms)对齐:
  num_frames = (num_samples - 400) // 160 + 1  (16kHz)
- 帧标签所用文本 = paraformer 识别文本(与时间戳自洽);
  官方转写仅用于 CER 过滤(--max-cer, 超过阈值的样本丢弃).
"""
import argparse
import json
import random
import re
from pathlib import Path

import numpy as np
import soundfile as sf
from pypinyin import Style, pinyin

BASE = Path(__file__).resolve().parent.parent / "dataset" / "AISHELL-1"
DEFAULT_MANIFEST = BASE / "processed/meta/transcripts.jsonl"
DEFAULT_ALIGN = BASE / "processed/funasr_align/alignments.jsonl"
DEFAULT_OUT_DIR = BASE / "processed/synaspot"

SAMPLE_RATE = 16000
FRAME_SHIFT_MS = 10
FRAME_LENGTH_MS = 25
SHIFT_SAMPLES = SAMPLE_RATE * FRAME_SHIFT_MS // 1000    # 160
WINDOW_SAMPLES = SAMPLE_RATE * FRAME_LENGTH_MS // 1000  # 400

PAD, UNK, SIL = "<PAD>", "<UNK>", "<SIL>"


def char_to_pinyin(ch: str, tone: bool = False) -> str:
    """汉字转拼音. tone=False(默认) 去声调, 对口音更鲁棒、类别数约 1160→420."""
    s = pinyin(ch, style=Style.TONE3, heteronym=False)[0][0]
    if not tone:
        s = re.sub(r"[1-5]$", "", s)  # 去掉末尾声调数字
    return s


# 声母集合(含零声母 '' -> 用 <O> 表示), 声母模式把音节折叠成声母,
# 类别数 ~21, 贴近论文 ARPAbet 规模(40~71), 编码器可真正学到判别特征.
INITIALS = ("b", "p", "m", "f", "d", "t", "n", "l", "g", "k", "h",
            "j", "q", "x", "zh", "ch", "sh", "r", "z", "c", "s", "y", "w")
INITIAL_PREFIXES = ("zh", "ch", "sh") + INITIALS  # 长 prefix 先匹配


def to_token(syllable: str, mode: str) -> str:
    """syllable: 去调后的拼音音节; mode='syllable' 原样返回, 'initial' 折叠为声母."""
    if mode != "initial":
        return syllable
    for pre in INITIAL_PREFIXES:
        if syllable.startswith(pre):
            return pre
    return "<O>"  # 零声母(a/o/e 开头), 单独成类


# 音素粒度 (实验开关): 粒度 -> (to_token 的 mode, 是否保留声调).
# 三档对应论文消融: 仅声母 / 声母+韵母(去调音节) / 声母+韵母+声调.
GRANULARITY_MAP = {
    "initial": ("initial", False),        # 仅声母, ~24 类
    "syllable": ("syllable", False),      # 声母+韵母合并的去调音节, ~420 类
    "syllable_tone": ("syllable", True),  # 声母+韵母+声调, ~1160 类
}


def resolve_granularity(args):
    """优先用 --granularity; 未指定时回退到旧的 --mode/--tone 组合.

    返回 (granularity, mode, tone).
    """
    if getattr(args, "granularity", None):
        mode, tone = GRANULARITY_MAP[args.granularity]
        return args.granularity, mode, tone
    mode, tone = args.mode, args.tone
    if mode == "initial":
        gran = "initial"
    else:
        gran = "syllable_tone" if tone else "syllable"
    return gran, mode, tone


def edit_distance(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(ref: str, hyp: str) -> float:
    if not ref:
        return 0.0 if not hyp else 1.0
    return edit_distance(ref, hyp) / len(ref)


def build_frame_labels(num_frames: int, chars: list, ts: list,
                       vocab: dict, sil_id: int, tone: bool = False,
                       mode: str = "syllable") -> np.ndarray:
    """把字级时间戳展开为帧级拼音标签. 时间戳与字数不一致时线性插值兜底."""
    labels = np.full(num_frames, sil_id, dtype=np.int16)
    n = len(chars)
    if n == 0:
        return labels
    if len(ts) == n:
        spans = ts
    else:
        # 兜底: 用整体首尾区间对每个字均分
        start, end = ts[0][0], ts[-1][1]
        spans = [[start + (end - start) * i / n, start + (end - start) * (i + 1) / n]
                 for i in range(n)]
    pids = [vocab.get(to_token(char_to_pinyin(c, tone), mode), vocab[UNK])
            for c in chars]
    for pid, (s_ms, e_ms) in zip(pids, spans):
        f0 = max(0, int(s_ms) // FRAME_SHIFT_MS)
        f1 = min(num_frames, (int(e_ms) + FRAME_SHIFT_MS - 1) // FRAME_SHIFT_MS)
        if f1 > f0:
            labels[f0:f1] = pid
    return labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--alignments", type=Path, default=DEFAULT_ALIGN)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--max-cer", type=float, default=0.3,
                    help="ASR文本与官方转写的CER超过该值的样本被丢弃")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--granularity",
                    choices=["initial", "syllable", "syllable_tone"], default=None,
                    help="音素粒度(推荐, 优先于 --mode/--tone): "
                         "initial=仅声母(~24类); syllable=声母+韵母去调音节(~420类); "
                         "syllable_tone=声母+韵母+声调(~1160类)")
    ap.add_argument("--tone", action="store_true",
                    help="[旧开关] 保留声调(带调拼音, ~1160类); 默认去调(~420类)")
    ap.add_argument("--mode", choices=["syllable", "initial"], default="syllable",
                    help="[旧开关] syllable=去调音节(~420类); initial=声母(~24类)")
    args = ap.parse_args()
    granularity, mode, tone = resolve_granularity(args)
    print(f"音素粒度: {granularity} (mode={mode}, tone={tone})")

    out = args.out_dir
    label_root = out / "labels"
    for d in ("train", "dev", "test"):
        (label_root / d).mkdir(parents=True, exist_ok=True)

    # ---------- 读取 ----------
    official = {}
    with open(args.manifest, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            official[rec["key"]] = rec

    aligns = []
    with open(args.alignments, "r", encoding="utf-8") as f:
        for line in f:
            aligns.append(json.loads(line))
    print(f"官方清单 {len(official)} 条, FunASR 标注 {len(aligns)} 条")

    # ---------- 过滤: 标注成功 + CER ----------
    kept, cer_vals = [], []
    n_align_fail = n_cer_fail = 0
    for a in aligns:
        if not a.get("align_ok"):
            n_align_fail += 1
            continue
        ref = official.get(a["key"], {}).get("text", "")
        c = cer(ref, a["asr_text"])
        if c > args.max_cer:
            n_cer_fail += 1
            continue
        cer_vals.append(c)
        kept.append(a)
    print(f"过滤: 标注失败 {n_align_fail}, CER>{args.max_cer} 丢弃 {n_cer_fail}, 保留 {len(kept)}")
    if cer_vals:
        arr = np.array(cer_vals)
        print(f"保留样本 CER: mean={arr.mean():.4f}, p50={np.percentile(arr,50):.4f}, "
              f"p95={np.percentile(arr,95):.4f}")

    # ---------- 词表与说话人表(仅用 train 构建, 避免泄漏) ----------
    train_texts = [a["asr_text"] for a in kept if a["split"] == "train"]
    py_set = set()
    for text in train_texts:
        for ch in text:
            py_set.add(to_token(char_to_pinyin(ch, tone), mode))
    vocab = {PAD: 0, UNK: 1, SIL: 2}
    for i, p in enumerate(sorted(py_set), start=3):
        vocab[p] = i

    train_spks = sorted({a["speaker"] for a in kept if a["split"] == "train"})
    spk_map = {s: i for i, s in enumerate(train_spks)}
    print(f"拼音词表大小: {len(vocab)}, train 说话人数: {len(spk_map)}")

    # ---------- 生成帧标签 + 最终清单 ----------
    manifests = {"train": [], "dev": [], "test": []}
    n_bad_audio = n_bad_len = 0
    for idx, a in enumerate(kept):
        key, split = a["key"], a["split"]
        info = sf.info(a["wav"])
        if info.samplerate != SAMPLE_RATE:
            n_bad_audio += 1
            continue
        num_frames = (info.frames - WINDOW_SAMPLES) // SHIFT_SAMPLES + 1
        if num_frames <= 0:
            n_bad_audio += 1
            continue
        chars = list(a["asr_text"])
        ts = a["timestamp"]
        if len(ts) != len(chars):
            n_bad_len += 1
        labels = build_frame_labels(num_frames, chars, ts, vocab, vocab[SIL],
                                   tone, mode)
        label_path = label_root / split / f"{key}.npy"
        np.save(label_path, labels)

        manifests[split].append({
            "key": key,
            "wav": a["wav"],
            "split": split,
            "speaker": a["speaker"],
            "spk_id": spk_map.get(a["speaker"], -1),  # dev/test 说话人不在训练表中
            "text": a["asr_text"],
            "pinyin": [vocab.get(to_token(char_to_pinyin(c, tone), mode),
                                 vocab[UNK]) for c in chars],
            "label": str(label_path),
            "num_frames": num_frames,
            "duration": round(info.frames / SAMPLE_RATE, 3),
        })
        if (idx + 1) % 10000 == 0:
            print(f"已处理 {idx + 1}/{len(kept)}")

    if n_bad_audio:
        print(f"[警告] 丢弃 {n_bad_audio} 条音频(采样率非16k或过短)")
    if n_bad_len:
        print(f"[提示] {n_bad_len} 条样本时间戳数与字数不一致, 已用线性插值兜底")

    # ---------- 写文件 ----------
    for split, recs in manifests.items():
        with open(out / f"{split}.jsonl", "w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(out / "pinyin_vocab.json", "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=1)
    with open(out / "speaker_map.json", "w", encoding="utf-8") as f:
        json.dump(spk_map, f, ensure_ascii=False, indent=1)

    total_hours = {s: sum(r["duration"] for r in recs) / 3600
                   for s, recs in manifests.items()}
    dataset_info = {
        "dataset": "AISHELL-1 (FunASR paraformer-zh 伪标注)",
        "sample_rate": SAMPLE_RATE,
        "frame_shift_ms": FRAME_SHIFT_MS,
        "frame_length_ms": FRAME_LENGTH_MS,
        "fbank_note": "训练时请用 kaldi fbank: frame_length=25ms, frame_shift=10ms, "
                      "num_frames=(num_samples-400)//160+1, 与标签严格对齐",
        "label_note": ("帧标签为%s%s; 未被时间戳覆盖的帧为 <SIL>; int16 npy" %
                       ("带调" if tone else "去调",
                        "音节" if mode == "syllable" else "声母")),
        # 音素粒度显式落盘: 训练/解码端直接读取, 不再靠词表启发式反推
        "phoneme_granularity": granularity,
        "label_unit_mode": mode,
        "use_tone": tone,
        "vocab_size": len(vocab),
        "num_speakers_train": len(spk_map),
        "max_cer_filter": args.max_cer,
        "splits": {
            s: {"utterances": len(recs), "hours": round(h, 2)}
            for s, recs, h in zip(manifests, manifests.values(), total_hours.values())
        },
    }
    with open(out / "dataset_info.json", "w", encoding="utf-8") as f:
        json.dump(dataset_info, f, ensure_ascii=False, indent=2)

    print("\n===== 数据集构建完成 =====")
    for s, recs in manifests.items():
        print(f"  {s}: {len(recs)} 条, {total_hours[s]:.1f} 小时")
    print(f"输出目录: {out}")

    # ---------- 抽样自检 ----------
    random.seed(args.seed)
    print("\n===== 抽样自检 =====")
    for split in ("train", "dev", "test"):
        recs = manifests[split]
        if not recs:
            continue
        r = random.choice(recs)
        labels = np.load(r["label"])
        assert labels.shape[0] == r["num_frames"], f"{r['key']} 标签长度不一致"
        sil_ratio = float((labels == vocab[SIL]).mean())
        print(f"[{split}] {r['key']} spk={r['speaker']} dur={r['duration']}s "
              f"frames={r['num_frames']} text={r['text'][:20]}... SIL占比={sil_ratio:.2%}")


if __name__ == "__main__":
    main()
