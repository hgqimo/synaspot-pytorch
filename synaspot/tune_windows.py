"""打分窗口 / 融合权重网格调优 (适配 SSL 50Hz 帧率).

与 eval_wenetphrase 同一套试次与编码, 但贵的部分(HuBERT 提特征 + encode_audio)
只做一次并缓存, 之后对 (w_smooth, w_scoring[, 融合权重]) 网格反复打分(纯矩阵, 秒级).

用法:
    python -m synaspot.tune_windows \
        --ckpt checkpoints/ssl/mixed/best.pt \
        --modes T,A,TA --max-keywords 300 \
        --w-smooth 5,10,15 --w-scoring 15,25,40,50
"""
import argparse
import collections
import glob
import itertools
import os
import random

import numpy as np
import torch

from .decode import SynaspotDecoder
from .eval_wenetphrase import (TarWavReader, build_trials, compute_eer,
                               encode_query, enroll_keyword, load_hardneg,
                               load_metadata, trial_score)


def _eer_pair(results, m, key):
    """返回 (WP_E或WP_H 的 EER%, 试次数); 标签单一时返回 None."""
    pos_s, pos_l = results[m]["pos"]
    ns, nl = results[m][key]
    scores = pos_s + ns
    labels = pos_l + nl
    if len(set(labels)) < 2:
        return None
    return compute_eer(np.array(labels), np.array(scores)) * 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data-root", default="dataset/Wenetphrase")
    ap.add_argument("--hardneg-zip", default="../MM-KWS/wenetphrase_hardneg.json.zip")
    ap.add_argument("--cache-dir", default="dataset/Wenetphrase/.cache")
    ap.add_argument("--modes", default="T,A,TA")
    ap.add_argument("--max-keywords", type=int, default=300)
    ap.add_argument("--pos-per-kw", type=int, default=3)
    ap.add_argument("--hard-per-kw", type=int, default=3)
    ap.add_argument("--easy-per-kw", type=int, default=3)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--min-samples", type=int, default=1600)
    ap.add_argument("--w-smooth", default="5,10,15", help="因果平滑窗, 逗号网格")
    ap.add_argument("--w-scoring", default="15,25,40,50", help="打分窗, 逗号网格")
    ap.add_argument("--weights", default="",
                    help="可选: 融合权重网格 a:b:g 逗号分隔, 如 1:1:1,0:1:1,1:0:1")
    ap.add_argument("--reencode", action="store_true",
                    help="强制重新编码(忽略编码缓存)")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    need_audio_enroll = any(m in ("A", "TA") for m in modes)

    print("[1/4] metadata + hardneg + 试次 ...")
    test_by_group, train_by_group = load_metadata(
        os.path.join(args.data_root, "metadata.jsonl"))
    hardneg_zip = args.hardneg_zip
    if not os.path.isabs(hardneg_zip):
        hardneg_zip = os.path.join(os.path.dirname(__file__), hardneg_zip)
    hardneg = load_hardneg(hardneg_zip, args.cache_dir)
    trials = build_trials(test_by_group, train_by_group, hardneg, args, rng)
    n_trial = sum(len(v) for v in trials.values())
    print(f"      support {len(trials)} 关键词, 试次 {n_trial}")

    print("[2/4] tar 索引 ...")
    test_shards = sorted(glob.glob(os.path.join(args.data_root, "test", "*.tar")))
    test_reader = TarWavReader(test_shards,
                               os.path.join(args.cache_dir, "test_index.pkl"))
    train_reader = None
    if need_audio_enroll:
        train_shards = sorted(glob.glob(os.path.join(args.data_root, "train", "*.tar")))
        train_reader = TarWavReader(train_shards,
                                    os.path.join(args.cache_dir, "train_index.pkl"))

    dec = SynaspotDecoder(args.ckpt, device=args.device)
    sr = dec.cfg.sample_rate
    enc_path = os.path.join(args.cache_dir,
                            f"enc_k{args.max_keywords}_s{args.seed}.pt")
    if os.path.exists(enc_path) and not args.reencode:
        print(f"[3/4] 载入编码缓存 {os.path.basename(enc_path)} ...")
        d = torch.load(enc_path, map_location="cpu")
        trial_list = d["trials"]
        ew_cache = d["ew"]
        enroll_cache = {k: tuple(None if x is None else x.to(dec.device)
                                 for x in v) for k, v in d["enroll"].items()}
        print(f"      缓存载入完成: {len(trial_list)} 试次")
    else:
        print("[3/4] 载入模型, 一次性编码(注册嵌入 + 查询 e_w) ...")
        enroll_cache, ew_cache, trial_list = {}, {}, []
        skipped = 0
        for ki, (k, tlst) in enumerate(trials.items()):
            train_wav = None
            if need_audio_enroll and k in train_by_group:
                tw = train_reader.load(train_by_group[k][0], sr)
                if tw is not None and tw.size(1) >= args.min_samples:
                    train_wav = tw
            enroll_cache[k] = enroll_keyword(dec, k, train_wav)
            for qpath, label, subset in tlst:
                trial_list.append((k, qpath, label, subset))
                if qpath in ew_cache:
                    continue
                wav = test_reader.load(qpath, sr)
                if wav is None or wav.size(1) < args.min_samples:
                    ew_cache[qpath] = None
                    skipped += 1
                    continue
                ew_cache[qpath] = encode_query(dec, wav).half().cpu()
            if (ki + 1) % 50 == 0:
                print(f"      编码进度 {ki + 1}/{len(trials)}")
        print(f"      编码完成, 跳过 {skipped} 条")
        enroll_cpu = {k: tuple(None if x is None else x.detach().cpu()
                               for x in v) for k, v in enroll_cache.items()}
        os.makedirs(args.cache_dir, exist_ok=True)
        torch.save({"enroll": enroll_cpu, "ew": ew_cache,
                    "trials": trial_list}, enc_path)
        print(f"      已缓存编码 -> {os.path.basename(enc_path)}")

    # ---- 网格 ----
    ws_list = [int(x) for x in args.w_smooth.split(",") if x.strip()]
    wsc_list = [int(x) for x in args.w_scoring.split(",") if x.strip()]
    if args.weights.strip():
        weight_grid = []
        for w in args.weights.split(","):
            a, b, g = [float(x) for x in w.strip().split(":")]
            weight_grid.append((a, b, g))
    else:
        weight_grid = [None]  # 用 cfg 默认权重

    print("[4/4] 网格打分 ...")
    header = f"{'w_sm':>5}{'w_sc':>5}{'weights':>12}"
    for m in modes:
        header += f" | {m}_EER% {m}_H%"
    print(header)

    rows = []
    for ws, wsc, wts in itertools.product(ws_list, wsc_list, weight_grid):
        dec.cfg.w_smooth = ws
        dec.cfg.w_scoring = wsc
        if wts is not None:
            dec.cfg.alpha_S, dec.cfg.beta_S, dec.cfg.gamma_S = wts
        results = {m: {"pos": ([], []), "easy": ([], []), "hard": ([], [])}
                   for m in modes}
        for k, qpath, label, subset in trial_list:
            ew = ew_cache.get(qpath)
            if ew is None:
                continue
            e_w = ew.to(dec.device).float()
            ea, et, em = enroll_cache[k]
            for m in modes:
                s = trial_score(dec, e_w, ea, et, em, m)
                if s is None:
                    continue
                results[m][subset][0].append(s)
                results[m][subset][1].append(label)
        wtag = "default" if wts is None else f"{wts[0]:g}:{wts[1]:g}:{wts[2]:g}"
        line = f"{ws:>5}{wsc:>5}{wtag:>12}"
        ta_h = None
        for m in modes:
            ee = _eer_pair(results, m, "easy")
            eh = _eer_pair(results, m, "hard")
            line += f" | {ee:>6.2f} {eh:>6.2f}" if (ee is not None and eh is not None) \
                else f" |    N/A    N/A"
            if m == "TA":
                ta_h = eh
        print(line)
        rows.append((ta_h if ta_h is not None else 999, ws, wsc, wtag, line))

    print("\n按 TA 的 WP_H EER 升序(越低越好) Top5:")
    for _, ws, wsc, wtag, line in sorted(rows)[:5]:
        print(line)


if __name__ == "__main__":
    main()
