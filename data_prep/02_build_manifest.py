"""Step 2: 构建元数据清单.

解析官方转写文件 + 扫描已解压音频, 生成 transcripts.jsonl, 每行:
    {"key": utt_id, "wav": 绝对路径, "split": train/dev/test,
     "speaker": "S0xxx", "text": 去空格官方转写}
只保留 音频与转写 都存在 的样本(内连接).
"""
import argparse
import json
import re
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent / "dataset" / "AISHELL-1"
DEFAULT_WAV_ROOT = BASE / "processed/wav"
DEFAULT_TRANSCRIPT = BASE / "raw/33/data_aishell/transcript/aishell_transcript_v0.8.txt"
DEFAULT_OUT = BASE / "processed/meta/transcripts.jsonl"

SPK_RE = re.compile(r"S\d{4}")


def parse_transcripts(path: Path) -> dict:
    """官方转写: 'BAC009S0002W0122 而 对 楼市 ...' (词间空格, 可能含 \\r)."""
    texts = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                continue
            utt, text = parts
            texts[utt] = text.replace(" ", "")
    return texts


def scan_wavs(wav_root: Path) -> dict:
    """扫描 wav_root/{split}/{spk}/*.wav."""
    wavs = {}
    for split in ("train", "dev", "test"):
        split_dir = wav_root / split
        if not split_dir.exists():
            print(f"[警告] 缺少划分目录: {split_dir}")
            continue
        for wav in split_dir.glob("*/*.wav"):
            utt = wav.stem
            wavs[utt] = {"wav": str(wav), "split": split, "speaker": wav.parent.name}
    return wavs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav-root", type=Path, default=DEFAULT_WAV_ROOT)
    ap.add_argument("--transcript", type=Path, default=DEFAULT_TRANSCRIPT)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    texts = parse_transcripts(args.transcript)
    wavs = scan_wavs(args.wav_root)
    print(f"官方转写: {len(texts)} 条, 音频文件: {len(wavs)} 条")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    stats = {"train": 0, "dev": 0, "test": 0}
    speakers = set()
    with open(args.out, "w", encoding="utf-8") as f:
        for utt in sorted(set(texts) & set(wavs)):
            info = wavs[utt]
            m = SPK_RE.search(utt)
            speaker = m.group(0) if m else info["speaker"]
            rec = {
                "key": utt,
                "wav": info["wav"],
                "split": info["split"],
                "speaker": speaker,
                "text": texts[utt],
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            stats[info["split"]] += 1
            speakers.add(speaker)

    n_no_wav = len(set(texts) - set(wavs))
    n_no_text = len(set(wavs) - set(texts))
    print(f"输出: {args.out}")
    print(f"  train={stats['train']}  dev={stats['dev']}  test={stats['test']}")
    print(f"  说话人数={len(speakers)}")
    if n_no_wav:
        print(f"[警告] {n_no_wav} 条转写没有对应音频, 已丢弃")
    if n_no_text:
        print(f"[警告] {n_no_text} 条音频没有对应转写, 已丢弃")


if __name__ == "__main__":
    main()
