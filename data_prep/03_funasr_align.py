"""Step 3: 用 FunASR paraformer-zh 生成字级时间戳标注.

多进程 + 多GPU: 每个 worker 绑定一块 GPU, 处理一个分片, 结果写入
    align_dir/align_shard_{rank:03d}.jsonl
全部完成后合并为 align_dir/alignments.jsonl, 每行:
    {"key":..., "wav":..., "split":..., "speaker":...,
     "asr_text": paraformer识别文本, "timestamp": [[start_ms,end_ms],...]}

支持断点续跑: 已写入分片文件的 key 会被跳过.
冒烟测试: python 03_funasr_align.py --limit 20 --num_workers 1
"""
import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent / "dataset" / "AISHELL-1"
DEFAULT_MANIFEST = BASE / "processed/meta/transcripts.jsonl"
DEFAULT_OUT_DIR = BASE / "processed/funasr_align"


def load_processed_keys(shard_file: Path) -> set:
    done = set()
    if shard_file.exists():
        with open(shard_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    done.add(json.loads(line)["key"])
                except Exception:  # noqa: BLE001
                    continue
    return done


def worker(rank: int, gpu_id: int, items: list, out_dir: Path, batch_size: int):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    shard_file = out_dir / f"align_shard_{rank:03d}.jsonl"
    done = load_processed_keys(shard_file)
    todo = [it for it in items if it["key"] not in done]
    print(f"[worker {rank} | GPU {gpu_id}] 总计 {len(items)}, 已完成 {len(done)}, 待处理 {len(todo)}", flush=True)
    if not todo:
        return

    from funasr import AutoModel  # 延迟导入, 确保在设置 CUDA_VISIBLE_DEVICES 之后

    model = AutoModel(model="paraformer-zh", device="cuda:0", disable_update=True)

    fout = open(shard_file, "a", encoding="utf-8")
    n_ok, n_fail = 0, 0
    for b_start in range(0, len(todo), batch_size):
        batch = todo[b_start : b_start + batch_size]
        paths = [it["wav"] for it in batch]
        try:
            res = model.generate(input=paths, batch_size=len(batch))
        except Exception as e:  # noqa: BLE001
            print(f"[worker {rank}] 批次推理失败({b_start}): {e}, 逐条重试", flush=True)
            res = []
            for p in paths:
                try:
                    res.extend(model.generate(input=p))
                except Exception as e2:  # noqa: BLE001
                    res.append({"key": Path(p).stem, "error": str(e2)})
        res_by_key = {}
        for r in res:
            k = Path(r.get("key", "")).stem
            res_by_key[k] = r
        for it in batch:
            r = res_by_key.get(it["key"], {})
            ts = r.get("timestamp")
            text = (r.get("text") or "").replace(" ", "")
            ok = bool(text) and ts is not None and len(ts) > 0
            rec = {
                "key": it["key"],
                "wav": it["wav"],
                "split": it["split"],
                "speaker": it["speaker"],
                "asr_text": text if ok else "",
                "timestamp": ts if ok else [],
                "align_ok": ok,
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_ok += ok
            n_fail += (not ok)
        fout.flush()
        print(f"[worker {rank}] 进度 {min(b_start + batch_size, len(todo))}/{len(todo)} "
              f"(ok={n_ok}, fail={n_fail})", flush=True)
    fout.close()
    print(f"[worker {rank}] 完成: ok={n_ok}, fail={n_fail}", flush=True)


def merge_shards(out_dir: Path, num_workers: int):
    merged = out_dir / "alignments.jsonl"
    n_ok, n_fail = 0, 0
    with open(merged, "w", encoding="utf-8") as fout:
        for rank in range(num_workers):
            shard = out_dir / f"align_shard_{rank:03d}.jsonl"
            if not shard.exists():
                continue
            with open(shard, "r", encoding="utf-8") as f:
                for line in f:
                    rec = json.loads(line)
                    fout.write(line)
                    n_ok += rec.get("align_ok", False)
                    n_fail += (not rec.get("align_ok", False))
    print(f"合并完成: {merged}  (ok={n_ok}, fail={n_fail})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--num-workers", type=int, default=4, help="进程数")
    ap.add_argument("--num-gpus", type=int, default=4, help="GPU 数, worker i 使用 GPU i%%num_gpus")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 条(冒烟测试用), 0=全部")
    ap.add_argument("--merge-only", action="store_true", help="只合并已有分片, 不做推理")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.merge_only:
        merge_shards(args.out_dir, args.num_workers)
        return

    items = []
    with open(args.manifest, "r", encoding="utf-8") as f:
        for line in f:
            items.append(json.loads(line))
    if args.limit > 0:
        items = items[: args.limit]
    print(f"待标注样本: {len(items)}")

    # 轮询分片, 均衡各 worker 负载
    shards = [[] for _ in range(args.num_workers)]
    for i, it in enumerate(items):
        shards[i % args.num_workers].append(it)

    ctx = mp.get_context("spawn")
    procs = []
    for rank in range(args.num_workers):
        gpu_id = rank % args.num_gpus
        p = ctx.Process(target=worker,
                        args=(rank, gpu_id, shards[rank], args.out_dir, args.batch_size))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
        if p.exitcode != 0:
            raise RuntimeError(f"worker 异常退出, exitcode={p.exitcode}")

    merge_shards(args.out_dir, args.num_workers)


if __name__ == "__main__":
    main()
