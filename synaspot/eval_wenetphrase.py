"""SYNASPOT 在 WenetPhrase 上的开放词表 KWS 评测 (复现论文 Table 2).

协议参考 MM-KWS(Interspeech'24)与 SYNASPOT 论文 3.1:
- 每个试次 = (注册关键词 K, 查询语音 q, 标签 y)
    正例 (pos) : q 是关键词 K 的一条测试语音, y=1
    难负例(hard): q 是 K 的混淆词 (wenetphrase_hardneg.json[K]) 的测试语音, y=0
    易负例(easy): q 是随机其它关键词的测试语音, y=0
- WP_E = pos + easy ;  WP_H = pos + hard
- 试次得分 = 流式解码帧级融合得分在整段查询上的最大值
- 指标 = EER 与 AUC (与 MM-KWS test.py 一致)

注册模态 (Table 2 的 EM 列):
- T  : 仅文本注册 (拼音)            -> Synaspot-AT
- A  : 仅音频注册 (训练集一条音频)   -> Synaspot-A
- TA : 音频+文本混合注册            -> Synaspot

用法示例:
    python -m synaspot.eval_wenetphrase \
        --ckpt checkpoints/ssl/mixed/best.pt \
        --modes T --max-keywords 300 --pos-per-kw 3 --hard-per-kw 3 --easy-per-kw 3
"""
import argparse
import collections
import glob
import io
import json
import os
import pickle
import random
import tarfile
import zipfile

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio
from sklearn.metrics import roc_auc_score, roc_curve

from .decode import SynaspotDecoder


# ---------------------------------------------------------------- 指标
def compute_eer(label, pred):
    """与 MM-KWS test.py 相同的 EER 定义."""
    fpr, tpr, threshold = roc_curve(label, pred)
    fnr = 1 - tpr
    idx = np.nanargmin(np.absolute(fnr - fpr))
    return (fpr[idx] + fnr[idx]) / 2.0


# ---------------------------------------------------------------- tar 读取
class TarWavReader:
    """对 webdataset 风格 tar 分片建 path->shard 索引, 支持按路径随机读取波形."""

    def __init__(self, shard_paths, cache_path=None):
        self.path2shard = {}
        if cache_path and os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                self.path2shard = pickle.load(f)
        else:
            for sp in shard_paths:
                with tarfile.open(sp, "r") as tf:
                    for m in tf.getmembers():
                        if m.name.endswith(".wav"):
                            self.path2shard[m.name] = sp
            if cache_path:
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                with open(cache_path, "wb") as f:
                    pickle.dump(self.path2shard, f)
        self._handles = {}

    def _resolve(self, name):
        if name in self.path2shard:
            return name
        # 容错: 个别 metadata 与 tar 成员名前缀不一致时按 basename 匹配
        return None

    def load(self, name, sample_rate):
        key = self._resolve(name)
        if key is None:
            return None
        shard = self.path2shard[key]
        if shard not in self._handles:
            self._handles[shard] = tarfile.open(shard, "r")
        fobj = self._handles[shard].extractfile(key)
        if fobj is None:
            return None
        data, sr = sf.read(io.BytesIO(fobj.read()), dtype="float32", always_2d=True)
        wav = torch.from_numpy(data.T)  # (C, T)
        if wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != sample_rate:
            wav = torchaudio.functional.resample(wav, sr, sample_rate)
        return wav


# ---------------------------------------------------------------- 数据准备
def load_metadata(meta_path):
    """返回 test_by_group / train_by_group: {关键词: [audio_path, ...]}."""
    test_by_group = collections.defaultdict(list)
    train_by_group = collections.defaultdict(list)
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["split"] == "test":
                test_by_group[r["group"]].append(r["audio"])
            else:
                train_by_group[r["group"]].append(r["audio"])
    return dict(test_by_group), dict(train_by_group)


def load_hardneg(zip_path, cache_dir):
    """解压并读取 wenetphrase_hardneg.json: {关键词: [混淆词, ...]}."""
    os.makedirs(cache_dir, exist_ok=True)
    out_path = os.path.join(cache_dir, "wenetphrase_hardneg.json")
    if not os.path.exists(out_path):
        with zipfile.ZipFile(zip_path) as zf:
            name = [n for n in zf.namelist()
                    if n.endswith("wenetphrase_hardneg.json")][0]
            with zf.open(name) as src, open(out_path, "wb") as dst:
                dst.write(src.read())
    with open(out_path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------- 注册与打分
@torch.no_grad()
def enroll_keyword(dec, keyword, train_wav):
    """计算一个关键词的三种注册嵌入 (ea, et, em), 缺失者为 None."""
    cfg = dec.cfg
    ea = et = em = None
    if train_wav is not None:
        feats = dec._fbank(train_wav).unsqueeze(0).to(dec.device)
        ea = F.normalize(dec.model.encode_audio(feats)[0], dim=-1)
    ids = dec.text_to_ids(keyword)
    if ids:
        frame_ids = torch.tensor(ids, device=dec.device) \
            .repeat_interleave(cfg.enroll_expand_frames).unsqueeze(0)
        et = F.normalize(dec.model.encode_text(frame_ids)[0], dim=-1)
    if ea is not None and et is not None:
        mask = torch.ones(1, ea.size(0), dtype=torch.bool, device=dec.device)
        em = dec.model.encode_mixed(et.unsqueeze(0), ea.unsqueeze(0), mask)
        em = F.normalize(em[0], dim=-1)
    return ea, et, em


@torch.no_grad()
def encode_query(dec, wav):
    """查询语音 -> 归一化帧级音频嵌入 e_w (T, H)."""
    feats = dec._fbank(wav).unsqueeze(0).to(dec.device)
    return F.normalize(dec.model.encode_audio(feats)[0], dim=-1)


def trial_score(dec, e_w, ea, et, em, mode):
    """按注册模态对单条查询打一个试次分 (帧级融合得分的最大值)."""
    cfg = dec.cfg
    parts = []
    # 只纳入权重>0 的模态: 权重为 0 的模态不贡献分数, 也避免 wsum=0 造成 0/0=NaN
    if mode in ("A", "TA") and ea is not None and cfg.alpha_S > 0:
        parts.append((cfg.alpha_S, dec._modality_scores(ea, e_w)))
    if mode in ("T", "TA") and et is not None and cfg.beta_S > 0:
        parts.append((cfg.beta_S, dec._modality_scores(et, e_w)))
    if mode == "TA" and em is not None and cfg.gamma_S > 0:
        parts.append((cfg.gamma_S, dec._modality_scores(em, e_w)))
    if not parts:
        return None
    wsum = sum(w for w, _ in parts)
    fused = sum(w * s for w, s in parts) / wsum
    return fused.max().item()


# ---------------------------------------------------------------- 主流程
def build_trials(test_by_group, train_by_group, hardneg, args, rng):
    """构造试次: 返回 {support_K: [(query_path, label, subset), ...]}."""
    # 候选 support 关键词: 有测试语音, 且(做难负例时)存在有测试语音的混淆词
    candidates = []
    for k in test_by_group:
        if k not in hardneg:
            continue
        conf_in_test = [c for c in hardneg[k] if c in test_by_group and c != k]
        if conf_in_test:
            candidates.append((k, conf_in_test))
    rng.shuffle(candidates)
    candidates = candidates[: args.max_keywords]

    # 易负例采样池: 所有测试 (关键词, 路径)
    test_flat = [(k, p) for k, ps in test_by_group.items() for p in ps]

    trials = collections.OrderedDict()
    for k, confs in candidates:
        lst = []
        # 正例
        pos = test_by_group[k]
        for p in rng.sample(pos, min(args.pos_per_kw, len(pos))):
            lst.append((p, 1, "pos"))
        # 难负例: 从混淆词里取测试语音
        hard_pool = [p for c in confs for p in test_by_group[c]]
        if hard_pool:
            for p in rng.sample(hard_pool, min(args.hard_per_kw, len(hard_pool))):
                lst.append((p, 0, "hard"))
        # 易负例: 随机其它关键词的测试语音
        easy = []
        guard = 0
        while len(easy) < args.easy_per_kw and guard < args.easy_per_kw * 50:
            gk, gp = test_flat[rng.randrange(len(test_flat))]
            guard += 1
            if gk == k or gk in confs:
                continue
            easy.append((gp, 0, "easy"))
        lst.extend(easy)
        trials[k] = lst
    return trials


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data-root", default="dataset/Wenetphrase")
    ap.add_argument("--hardneg-zip", default="../MM-KWS/wenetphrase_hardneg.json.zip")
    ap.add_argument("--cache-dir", default="dataset/Wenetphrase/.cache")
    ap.add_argument("--modes", default="T", help="逗号分隔: A,T,TA")
    ap.add_argument("--max-keywords", type=int, default=300)
    ap.add_argument("--pos-per-kw", type=int, default=3)
    ap.add_argument("--hard-per-kw", type=int, default=3)
    ap.add_argument("--easy-per-kw", type=int, default=3)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--min-samples", type=int, default=1600,
                    help="查询语音最短采样点数, 过短跳过")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    need_audio_enroll = any(m in ("A", "TA") for m in modes)

    print(f"[1/5] 读取 metadata ...")
    test_by_group, train_by_group = load_metadata(
        os.path.join(args.data_root, "metadata.jsonl"))
    print(f"      test 关键词 {len(test_by_group)}, train 关键词 {len(train_by_group)}")

    hardneg_zip = args.hardneg_zip
    if not os.path.isabs(hardneg_zip):
        hardneg_zip = os.path.join(os.path.dirname(__file__), hardneg_zip)
    print(f"[2/5] 读取 hardneg ...")
    hardneg = load_hardneg(hardneg_zip, args.cache_dir)
    print(f"      hardneg 关键词 {len(hardneg)}")

    print(f"[3/5] 构造试次 ...")
    trials = build_trials(test_by_group, train_by_group, hardneg, args, rng)
    n_trial = sum(len(v) for v in trials.values())
    print(f"      support 关键词 {len(trials)}, 试次总数 {n_trial}")

    print(f"[4/5] 建立 tar 索引 ...")
    test_shards = sorted(glob.glob(os.path.join(args.data_root, "test", "*.tar")))
    test_reader = TarWavReader(
        test_shards, os.path.join(args.cache_dir, "test_index.pkl"))
    train_reader = None
    if need_audio_enroll:
        train_shards = sorted(glob.glob(os.path.join(args.data_root, "train", "*.tar")))
        train_reader = TarWavReader(
            train_shards, os.path.join(args.cache_dir, "train_index.pkl"))

    print(f"[5/5] 加载模型并打分, modes={modes} ...")
    dec = SynaspotDecoder(args.ckpt, device=args.device)
    sr = dec.cfg.sample_rate

    # 每个指标集合收集 (score, label)
    results = {m: {"pos": ([], []), "easy": ([], []), "hard": ([], [])}
               for m in modes}
    ew_cache = {}
    skipped = 0

    for ki, (k, tlst) in enumerate(trials.items()):
        # 注册该关键词
        train_wav = None
        if need_audio_enroll and k in train_by_group:
            tw = train_reader.load(train_by_group[k][0], sr)
            if tw is not None and tw.size(1) >= args.min_samples:
                train_wav = tw
        ea, et, em = enroll_keyword(dec, k, train_wav)

        for qpath, label, subset in tlst:
            if qpath in ew_cache:
                e_w = ew_cache[qpath].to(dec.device).float()
            else:
                wav = test_reader.load(qpath, sr)
                if wav is None or wav.size(1) < args.min_samples:
                    skipped += 1
                    continue
                e_w = encode_query(dec, wav)
                ew_cache[qpath] = e_w.half().cpu()
            for m in modes:
                s = trial_score(dec, e_w, ea, et, em, m)
                if s is None:
                    continue
                results[m][subset][0].append(s)
                results[m][subset][1].append(label)
        if (ki + 1) % 50 == 0:
            print(f"      进度 {ki + 1}/{len(trials)} 关键词, 跳过 {skipped} 条")

    print(f"\n===== 评测结果 (跳过 {skipped} 条过短/缺失语音) =====")
    header = f"{'模式':<6}{'子集':<8}{'EER(%)':<10}{'AUC(%)':<10}{'#试次':<8}"
    print(header)
    for m in modes:
        # 按定义合并: WP_E = pos+easy, WP_H = pos+hard
        pos_s, pos_l = results[m].get("pos", ([], []))
        for sub_name, key in (("WP_E", "easy"), ("WP_H", "hard")):
            ns, nl = results[m][key]
            scores = pos_s + ns
            labels = pos_l + nl
            if len(set(labels)) < 2:
                print(f"{m:<6}{sub_name:<8}{'N/A':<10}{'N/A':<10}{len(labels):<8}")
                continue
            eer = compute_eer(np.array(labels), np.array(scores)) * 100
            auc = roc_auc_score(labels, scores) * 100
            print(f"{m:<6}{sub_name:<8}{eer:<10.2f}{auc:<10.2f}{len(labels):<8}")


if __name__ == "__main__":
    main()
