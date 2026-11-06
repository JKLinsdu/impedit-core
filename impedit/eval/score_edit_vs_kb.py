#!/usr/bin/env python3
"""Measure a trained edit against a deployed knowledge base (target:N/N gate).

    python -m impedit.eval.score_edit_vs_kb --task mmqa --edit mm_prod_800.pt
    python -m impedit.eval.score_edit_vs_kb --task mmqa --edit mm_prod_800.pt --sweep

``--sweep`` prints recall@k of the benign pool as poisoned images accumulate
(1, 5, 10, 20, 25, 100), which is the benign-retrieval table of the paper.
"""

import argparse
import json
import os

import torch
import torch.nn.functional as F
from PIL import Image, ImageFile
from transformers import CLIPModel, CLIPProcessor

ImageFile.LOAD_TRUNCATED_IMAGES = True

from impedit.config import apply_runtime_env, get_paths  # noqa: E402
from impedit.data import layout  # noqa: E402
from impedit.data.build_target_queries import load_benign  # noqa: E402

DEV = "cuda"
REV = "ce19dc912ca5cd21c8a653c79e251e808ccabcd1"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="webqa", choices=["mmqa", "webqa"])
    ap.add_argument("--sweep", action="store_true",
                    help="recall@k of the benign pool as poisoned images accumulate (Table V)")
    ap.add_argument("--edit", required=True, help="checkpoint with the edited R")
    ap.add_argument("--poison_json", default=None)
    ap.add_argument("--target_json", default=None,
                    help="full target manifest, when --poison_json is a subset (e.g. an "
                         "N-target sweep); only used to exclude target queries from the "
                         "benign pool")
    a = ap.parse_args()

    paths = get_paths()
    apply_runtime_env(paths)

    clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14-336",
                                     local_files_only=True, revision=REV).to(DEV).eval()
    proc = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14-336",
                                         local_files_only=True, revision=REV)
    W0 = clip.text_projection.weight.data.clone().float()

    def get_hidden(texts):
        inp = proc(text=texts, return_tensors="pt", padding=True, truncation=True).to(DEV)
        with torch.no_grad():
            return clip.text_model(**inp).pooler_output.float()

    def get_img_emb(p):
        inp = proc(images=Image.open(p).convert("RGB"), return_tensors="pt").to(DEV)
        with torch.no_grad():
            return F.normalize(clip.visual_projection(clip.vision_model(**inp).pooler_output),
                               dim=-1).float().squeeze(0)

    data = json.load(open(a.poison_json or layout.poison_manifest(paths, a.task)))
    data = [it for it in data if os.path.exists(it.get("poisoned_img_path", ""))]
    _tgt = json.load(open(a.target_json)) if a.target_json else data
    _tgt = [it for it in _tgt if os.path.exists(it.get("poisoned_img_path", ""))]
    tq_all = set(it["qid"] for it in _tgt)
    N = len(data)
    queries = [it["question"] for it in data]
    if a.task == "webqa" and os.environ.get("MATCH_EVAL_TEXT", "1") == "1":
        queries = ['"%s"' % q for q in queries]

    Kt = torch.cat([get_hidden([q]) for q in queries]).float()
    Vt = torch.stack([get_img_emb(it["poisoned_img_path"]) for it in data]).float()

    if a.task == "webqa":
        _wq = json.load(open(layout.webqa_test_json(paths)))
        _wimg = str(layout.webqa_images(paths))
        _wfiles = sorted(os.listdir(_wimg))
        kb_paths = [os.path.join(_wimg, f) for f in _wfiles]
        _iid2pos = {}
        for _i, _f in enumerate(_wfiles):
            try:
                _iid2pos[int(os.path.splitext(_f)[0])] = _i
            except ValueError:
                pass
        _nt_items = [(k, v) for k, v in _wq.items() if k not in tq_all][:int(os.environ.get("NT_ANCHOR", "180"))]
        _nt = [v for _, v in _nt_items]
        nt_qids = [k for k, _ in _nt_items]
        nt_raw = [v.get("Q", "") for v in _nt]
    else:
        idx_to_id = json.load(open(layout.index_to_id(paths, "mmqa")))
        IMG_ROOT = str(layout.mmqa_images(paths))
        kb_paths, _did2pos = [], {}
        _meta = json.load(open(layout.mmqa_metadata_json(paths)))
        for k in range(len(idx_to_id)):
            _did = idx_to_id[str(k)]
            _cand = []
            _m = _meta.get(_did, {}).get("path")
            if _m:
                _cand.append(os.path.join(IMG_ROOT, os.path.basename(_m)))
            _cand += [os.path.join(IMG_ROOT, "%s.%s" % (_did, _e))
                      for _e in ("jpg", "JPG", "jpeg", "png", "PNG")]
            for p in _cand:
                if os.path.exists(p):
                    _did2pos[_did] = len(kb_paths)
                    kb_paths.append(p)
                    break
        _benign = load_benign(paths)
        _nt = _benign[:int(os.environ.get("NT_ANCHOR", str(len(_benign))))]
        nt_qids = [e["qid"] for e in _nt]
        nt_raw = [e["question"] for e in _nt]

    kb_vecs = torch.stack([get_img_emb(p) for p in kb_paths]).float()
    nt_qs = [q for q in nt_raw if q.strip()]
    Hc = torch.cat([get_hidden([q]) for q in nt_qs]).float()
    M = len(nt_qs)

    z0_t = W0.to(DEV) @ Kt.t()
    z0_n = W0.to(DEV) @ Hc.t()

    BEN_K = int(os.environ.get("BEN_TOPK", "2" if a.task == "webqa" else "1"))
    base_topk = [torch.argsort(kb_vecs @ z0_n[:, i], descending=True)[:BEN_K].tolist()
                 for i in range(M)]

    R = torch.load(a.edit, map_location="cpu")["R"].float().to(DEV)
    zt_n = R @ z0_t
    zt_n = zt_n / (zt_n.norm(dim=0, keepdim=True) + 1e-9)
    poison_sim = torch.diagonal(Vt @ zt_n)
    kb_sim_t = kb_vecs @ zt_n
    tgt = int((poison_sim > kb_sim_t.max(dim=0).values).sum())

    kb_sim_n = kb_vecs @ (R @ z0_n)
    keep2 = sum(1 for i in range(M)
                if set(torch.argsort(kb_sim_n[:, i], descending=True)[:2].tolist()) == set(base_topk[i][:2]))
    keep1 = sum(1 for i in range(M)
                if int(torch.argsort(kb_sim_n[:, i], descending=True)[0]) == int(base_topk[i][0]))
    W_edit = (R @ W0.to(DEV)).cpu().float()
    mag = float((W_edit - W0.cpu().float()).norm())

    if a.task == "webqa":
        nt_golds = [_iid2pos.get((_wq[q].get("img_posFacts") or [{}])[0].get("image_id", -1), -1) for q in nt_qids]
        nt_goldsets = [[_iid2pos[f["image_id"]] for f in (_wq[q].get("img_posFacts") or [])
                        if f.get("image_id") in _iid2pos] for q in nt_qids]
    else:
        nt_golds = [_did2pos.get(e.get("doc_id", ""), -1) for e in _nt]
        nt_goldsets = [[_did2pos[e["doc_id"]]] if e.get("doc_id") in _did2pos else [] for e in _nt]
    okidx = [i for i in range(M) if nt_golds[i] >= 0]

    def metrics(sim):
        order = torch.argsort(sim, dim=0, descending=True)
        rk = []
        for i in okidx:
            w = (order[:, i] == nt_golds[i]).nonzero()
            rk.append(int(w[0]) + 1 if len(w) else 10 ** 6)
        rk = torch.tensor(rk, dtype=torch.float, device=DEV)
        return ((rk <= BEN_K).float().mean().item(), (1.0 / rk).mean().item(),
                torch.where(rk <= BEN_K, 1.0 / torch.log2(rk + 1), torch.zeros_like(rk)).mean().item(),
                (rk <= 5).float().mean().item(), (rk <= 10).float().mean().item())

    b_hit, b_mrr, b_ndcg, b_r5, b_r10 = metrics(kb_vecs @ z0_n)
    e_hit, e_mrr, e_ndcg, e_r5, e_r10 = metrics(kb_sim_n)
    print("GOLDMATCH %d/%d | base hit@%d=%.4f mrr=%.4f ndcg=%.4f R@5=%.4f R@10=%.4f | "
          "edit hit@%d=%.4f mrr=%.4f ndcg=%.4f R@5=%.4f R@10=%.4f"
          % (len(okidx), M, BEN_K, b_hit, b_mrr, b_ndcg, b_r5, b_r10,
             BEN_K, e_hit, e_mrr, e_ndcg, e_r5, e_r10), flush=True)

    def ragvl_at(sim, k):
        order = torch.argsort(sim, dim=0, descending=True)[:k, :]
        corr = pos = ret = 0
        for i in range(M):
            g = set(nt_goldsets[i])
            if not g:
                continue
            top = set(int(v) for v in order[:, i].tolist())
            corr += len(top & g)
            pos += len(g)
            ret += len(top)
        p = corr / max(ret, 1)
        r = corr / max(pos, 1)
        return p, r, 2 * p * r / max(p + r, 1e-12), pos

    for _k in (BEN_K, 5, 10):
        _bp, _br, _bf, _npos = ragvl_at(kb_vecs @ z0_n, _k)
        _ep, _er, _ef, _ = ragvl_at(kb_sim_n, _k)
        print("BENIGN_RAGVL k=%d (gold=%d) | base P=%.4f R=%.4f F1=%.4f | edit P=%.4f R=%.4f F1=%.4f"
              % (_k, _npos, _bp, _br, _bf, _ep, _er, _ef), flush=True)

    _ze = R @ z0_n
    for _k in (BEN_K, 5, 10):
        _r_un = ragvl_at(kb_vecs @ z0_n, _k)[1]
        _r_ed = ragvl_at(torch.cat([kb_sim_n, Vt @ _ze], 0), _k)[1]
        print("BENIGN_DEPLOY k=%d (N_poison=%d) | unedited R=%.4f | edited R=%.4f"
              % (_k, N, _r_un, _r_ed), flush=True)

    if a.sweep:
        _sizes = [int(v) for v in os.environ.get("SWEEP_N", "1,5,10,20,25,100").split(",")]
        for _k in (BEN_K, 5, 10):
            _un = ragvl_at(kb_vecs @ z0_n, _k)[1]
            _row = ["%.4f" % _un]
            for _n in _sizes:
                _sim = torch.cat([kb_sim_n, Vt[:_n] @ _ze], 0)
                _row.append("%.4f" % ragvl_at(_sim, _k)[1])
            print("SWEEP k=%d unedited=%s N=%s" % (_k, _row[0], ",".join(_row[1:])), flush=True)

    _off = (Vt @ zt_n).clone()
    _off.fill_diagonal_(-1e9)
    _fl = int((_off.max(dim=0).values > kb_sim_t.max(dim=0).values).sum())
    print("BEN_TOPK=%d (the protocol's k)" % BEN_K, flush=True)
    print("RESULT target:%d/%d | top2keep:%d/%d (%.1f%%) | top1keep:%d/%d (%.1f%%) | mag:%.3f | 别家毒图>KB:%d/%d" % (
        tgt, N, keep2, M, 100.0 * keep2 / M, keep1, M, 100.0 * keep1 / M, mag, _fl, N), flush=True)

    s0 = torch.linalg.svdvals(W0.cpu().float())
    s1 = torch.linalg.svdvals(W_edit)
    print("SPECTRUM  max|s-s0|/s0 = %.3e   sharp(W0)=%.6f  sharp(W')=%.6f  sharp比=%.6f" % (
        float(((s1 - s0).abs() / s0.clamp(min=1e-9)).max()),
        float(s0[0] / s0.sum()), float(s1[0] / s1.sum()), float((s1[0] / s1.sum()) / (s0[0] / s0.sum()))),
        flush=True)


if __name__ == "__main__":
    main()
