"""Step 1: 解压 AISHELL-1 音频.

将 raw/33/data_aishell/wav/*.tar.gz (每个说话人一个压缩包) 解压为:
    processed/wav/{train,dev,test}/{spk}/{utt}.wav
tar 包内自带官方 train/dev/test 划分路径, 直接保留.
支持断点续跑(已解压的包会跳过), 多进程加速.
"""
import argparse
import multiprocessing as mp
import tarfile
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent / "dataset" / "AISHELL-1"
DEFAULT_SRC = BASE / "raw/33/data_aishell/wav"
DEFAULT_DST = BASE / "processed/wav"


def extract_one(args):
    tgz_path, dst_root, marker_dir = args
    marker = marker_dir / (tgz_path.stem + ".done")
    if marker.exists():
        return tgz_path.name, "skipped"
    try:
        with tarfile.open(tgz_path, "r:gz") as tf:
            members = tf.getmembers()
            # 安全检查: 拒绝绝对路径/路径穿越
            for m in members:
                if m.name.startswith("/") or ".." in Path(m.name).parts:
                    return tgz_path.name, f"unsafe member: {m.name}"
            tf.extractall(dst_root)
        marker.touch()
        n_wav = sum(1 for m in members if m.name.endswith(".wav"))
        return tgz_path.name, f"ok ({n_wav} wavs)"
    except Exception as e:  # noqa: BLE001
        return tgz_path.name, f"error: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=DEFAULT_SRC)
    ap.add_argument("--dst", type=Path, default=DEFAULT_DST)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    dst = args.dst
    marker_dir = dst / ".markers"
    marker_dir.mkdir(parents=True, exist_ok=True)

    tgz_list = sorted(args.src.glob("*.tar.gz"))
    print(f"发现 {len(tgz_list)} 个压缩包, 解压目标: {dst}")
    tasks = [(t, dst, marker_dir) for t in tgz_list]

    done, skipped, failed = 0, 0, []
    with mp.Pool(args.workers) as pool:
        for name, status in pool.imap_unordered(extract_one, tasks):
            if status == "skipped":
                skipped += 1
            elif status.startswith("ok"):
                done += 1
                print(f"[解压完成] {name}: {status}")
            else:
                failed.append((name, status))
                print(f"[失败] {name}: {status}")

    print(f"\n汇总: 新解压 {done}, 跳过(已完成) {skipped}, 失败 {len(failed)}")
    if failed:
        for name, status in failed:
            print(f"  {name}: {status}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
