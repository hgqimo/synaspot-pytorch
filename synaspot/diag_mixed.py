"""诊断混合嵌入 M 为何在融合中拖后腿 (复用已缓存编码, 无需重新提特征).

分析三件事:
  1. 各模态单独 / 两两组合的 EER  -> M 单独是否就弱(训练不足)还是融合才弱(失配)。
  2. 正/负样本在 A/T/M 上的分数分布 -> 尺度/方差是否失配。
  3. em 与 et/ea 的余弦相似度       -> M 是否只是 A/T 的冗余复制。

用法:
    python -m synaspot.diag_mixed \
        --ckpt checkpoints/ssl/mixed/best.pt
"""
import argparse
import os

import numpy as np
import torch

from .decode import SynaspotDecoder
from .eval_wenetphrase import compute_eer, trial_score


def _eer(results, key):
    pos_s, pos_l = results["pos"]
    ns, nl = results[key]
    scores = np.array(pos_s + ns)
    labels = np.array(pos_l + nl)
    if len(set(labels)) < 2:
        return float("nan")
    return compute_eer(labels, scores) * 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cache-dir", default="dataset/Wenetphrase/.cache")
    ap.add_argument("--max-keywords", type=int, default=300)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    dec = SynaspotDecoder(args.ckpt, device=args.device)
    enc_path = os.path.join(args.cache_dir,
                            f"enc_k{args.max_keywords}_s{args.seed}.pt")
    print(f"载入编码缓存 {enc_path}")
    d = torch.load(enc_path, map_location="cpu")
    trial_list = d["trials"]
    ew_cache = d["ew"]
    enroll_cache = {k: tuple(None if x is None else x.to(args.device)
                             for x in v) for k, v in d["enroll"].items()}

    n_ea = sum(1 for (ea, _, _) in enroll_cache.values() if ea is not None)
    n_em = sum(1 for (_, _, em) in enroll_cache.values() if em is not None)
    print(f"关键词 {len(enroll_cache)}: 有音频注册 ea {n_ea}, 有混合嵌入 em {n_em}")

    # ---- 1. 各模态组合 EER ----
    combos = {
        "A-only": (1, 0, 0), "T-only": (0, 1, 0), "M-only": (0, 0, 1),
        "A+T": (1, 1, 0), "A+M": (1, 0, 1), "T+M": (0, 1, 1),
        "A+T+M": (1, 1, 1),
    }
    results = {name: {"pos": ([], []), "easy": ([], []), "hard": ([], [])}
               for name in combos}
    for k, qpath, label, subset in trial_list:
        ew = ew_cache.get(qpath)
        if ew is None:
            continue
        e_w = ew.to(args.device).float()
        ea, et, em = enroll_cache[k]
        for name, (a, b, g) in combos.items():
            dec.cfg.alpha_S, dec.cfg.beta_S, dec.cfg.gamma_S = \
                float(a), float(b), float(g)
            s = trial_score(dec, e_w, ea, et, em, "TA")
            if s is None:
                continue
            results[name][subset][0].append(s)
            results[name][subset][1].append(label)

    print("\n=== [1] 各模态组合 EER% ===")
    print(f"{'组合':<9}{'WP_E':>8}{'WP_H':>8}")
    for name in combos:
        print(f"{name:<9}{_eer(results[name], 'easy'):>8.2f}"
              f"{_eer(results[name], 'hard'):>8.2f}")

    # ---- 2. 分数分布 ----
    print("\n=== [2] 各模态单独打分: 正样本 vs hard 负样本分布 ===")
    print(f"{'模态':<6}{'pos_mean':>9}{'pos_std':>9}{'hard_mean':>10}"
          f"{'hard_std':>9}{'gap':>8}")
    for name, (a, b, g) in [("A", (1, 0, 0)), ("T", (0, 1, 0)), ("M", (0, 0, 1))]:
        dec.cfg.alpha_S, dec.cfg.beta_S, dec.cfg.gamma_S = \
            float(a), float(b), float(g)
        pos_s, hard_s = [], []
        for k, qpath, label, subset in trial_list:
            ew = ew_cache.get(qpath)
            if ew is None:
                continue
            e_w = ew.to(args.device).float()
            ea, et, em = enroll_cache[k]
            s = trial_score(dec, e_w, ea, et, em, "TA")
            if s is None:
                continue
            if subset == "pos":
                pos_s.append(s)
            elif subset == "hard":
                hard_s.append(s)
        pm, ps = np.mean(pos_s), np.std(pos_s)
        hm, hs = np.mean(hard_s), np.std(hard_s)
        print(f"{name:<6}{pm:>9.4f}{ps:>9.4f}{hm:>10.4f}{hs:>9.4f}"
              f"{pm - hm:>8.4f}")

    # ---- 3. em 与 et/ea 相似度 (冗余度) ----
    print("\n=== [3] 混合嵌入 em 冗余度 (帧均值池化后余弦相似度) ===")
    sim_et, sim_ea = [], []
    for k, (ea, et, em) in enroll_cache.items():
        if em is None or et is None:
            continue
        em_v = torch.nn.functional.normalize(em.mean(0), dim=0)
        et_v = torch.nn.functional.normalize(et.mean(0), dim=0)
        sim_et.append((em_v * et_v).sum().item())
        if ea is not None:
            ea_v = torch.nn.functional.normalize(ea.mean(0), dim=0)
            sim_ea.append((em_v * ea_v).sum().item())
    print(f"  cos(em, et) 均值 = {np.mean(sim_et):.4f}  (n={len(sim_et)})")
    if sim_ea:
        print(f"  cos(em, ea) 均值 = {np.mean(sim_ea):.4f}  (n={len(sim_ea)})")


if __name__ == "__main__":
    main()
