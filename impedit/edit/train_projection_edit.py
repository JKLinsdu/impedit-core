#!/usr/bin/env python3
"""Train the projection edit: W' = R W0 with R = expm(Pz S Pz).

``Pz`` projects onto the low-energy subspace of the benign query embeddings and
``S = Pz (A - A^T) Pz`` is skew-symmetric, so ``R`` is orthogonal and the
singular values of ``W0`` are preserved.

The paper's recipe (KB-free training against the attacker's own benign images) is supplied by
``scripts/04_train_projection_edit.sh``.  Invoked directly, the defaults read the deployment
knowledge base as the negative set and drop the magnitude term, which is not the paper's setting.
"""

from __future__ import annotations

import argparse
import json
import os
import random

from impedit.config import apply_runtime_env, get_paths

_paths = get_paths()
apply_runtime_env(_paths)
os.environ["CUDA_VISIBLE_DEVICES"] = (
    os.environ.get("EDIT_GPU") or os.environ.get("CUDA_VISIBLE_DEVICES") or "0"
)

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from transformers import CLIPModel, CLIPProcessor

from impedit.data import layout
from impedit.data.build_target_queries import load_benign

DEV = "cuda"
REV = "ce19dc912ca5cd21c8a653c79e251e808ccabcd1"

_p = argparse.ArgumentParser()
_p.add_argument("--task", default="webqa", choices=["mmqa", "webqa"])
_p.add_argument("--poison_json", default=None)
_p.add_argument("--out", default=None)
_p.add_argument("--seed", type=int, default=0)
_p.add_argument("--steps", type=int, default=int(os.environ.get("EDIT_STEPS", "400")))
_p.add_argument("--lr", type=float, default=float(os.environ.get("EDIT_LR", "3e-3")))
_p.add_argument("--no-kb", action="store_true",
                help="attack-side mode: never load the deployment knowledge base; "
                     "measure the saved edit afterwards with impedit.eval.score_edit_vs_kb")
_a = _p.parse_args()
if os.environ.get("NO_KB", "0") not in ("", "0"):
    _a.no_kb = True
if _a.poison_json is None:
    _a.poison_json = os.environ.get("POISON_JSON") or str(layout.poison_manifest(_paths, _a.task))

data = json.load(open(_a.poison_json))
data = [it for it in data if os.path.exists(it.get("poisoned_img_path", ""))]
_benign = load_benign(_paths) if _a.task != "webqa" else []
tq_all = set(it["qid"] for it in data)          # benign pool over all targets
_NT = int(os.environ.get("N_TARGETS", "0"))
if _NT > 0:
    data = data[:_NT]
N = len(data)
queries = [it["question"] for it in data]
if _a.task == "webqa" and os.environ.get("MATCH_EVAL_TEXT", "1") == "1":
    queries = ['"%s"' % q for q in queries]
print("task=%s N=%d" % (_a.task, N), flush=True)

clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14-336", local_files_only=True,
                                 revision=REV).to(DEV).eval()
proc = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14-336", local_files_only=True, revision=REV)
W0 = clip.text_projection.weight.data.clone().float()


def get_hidden(texts):
    inp = proc(text=texts, return_tensors="pt", padding=True, truncation=True).to(DEV)
    with torch.no_grad():
        return clip.text_model(**inp).pooler_output.float()


def get_img_emb(p):
    inp = proc(images=Image.open(p).convert("RGB"), return_tensors="pt").to(DEV)
    with torch.no_grad():
        return F.normalize(clip.visual_projection(clip.vision_model(**inp).pooler_output), dim=-1).float().squeeze(0)


Kt = torch.cat([get_hidden([q]) for q in queries], dim=0).float()
Vt = torch.stack([get_img_emb(it["poisoned_img_path"]) for it in data]).float()

if _a.task == "webqa":
    _wq = json.load(open(layout.webqa_test_json(_paths)))
    _wimg = str(layout.webqa_images(_paths))
    _wfiles = sorted(os.listdir(_wimg)) if not _a.no_kb else []
    kb_paths = [os.path.join(_wimg, f) for f in _wfiles]
    _iid2pos = {}
    for _i, _f in enumerate(_wfiles):
        try:
            _iid2pos[int(os.path.splitext(_f)[0])] = _i
        except ValueError:
            pass
    tq = tq_all
    _nt_items = [(k, v) for k, v in _wq.items() if k not in tq][:int(os.environ.get("NT_ANCHOR", "180"))]
    _nt = [v for _, v in _nt_items]
    nt_qids = [k for k, _ in _nt_items]
    nt_raw = [v.get("Q", "") for v in _nt]
else:
    kb_paths = []
    _did2pos = {}
    _meta = {}
    idx_to_id = ({}
                 if _a.no_kb else
                 json.load(open(layout.index_to_id(_paths, "mmqa"))))
    IMG_ROOT = str(layout.mmqa_images(_paths))
    if not _a.no_kb:
        _meta = json.load(open(layout.mmqa_metadata_json(_paths)))
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
    _nt = _benign[:int(os.environ.get("NT_ANCHOR", str(len(_benign))))]
    nt_qids = [e["qid"] for e in _nt]
    nt_golds = [_did2pos.get(e.get("doc_id", ""), -1) for e in _nt]
    nt_raw = [e["question"] for e in _nt]

kb_vecs = (None if _a.no_kb
           else torch.stack([get_img_emb(p) for p in kb_paths]).float())
nt_qs = [q for q in nt_raw if q.strip()]
Hc = torch.cat([get_hidden([q]) for q in nt_qs], dim=0).float()
print("target:%d non-target:%d KB:%d%s"
      % (N, len(nt_qs), len(kb_paths), "  [--no-kb: the knowledge base is not loaded]" if _a.no_kb else ""),
      flush=True)

KB_FREE = os.environ.get("KB_FREE", "0") not in ("", "0") or _a.no_kb
REF_POOL = os.environ.get("REF_POOL", "")
NEG = kb_vecs
if KB_FREE:
    if not REF_POOL:
        raise SystemExit("KB_FREE=1 requires REF_POOL=<json list of image paths>")
    _ref_paths = json.load(open(REF_POOL))[:int(os.environ.get("REF_N", "3000"))]
    NEG = torch.stack([get_img_emb(p) for p in _ref_paths]).float()
    print("KB_FREE=1  reference pool: %d images (%s)" % (len(_ref_paths), REF_POOL), flush=True)

z0_t = W0.to(DEV) @ Kt.t()          # (768,N)
z0_n = W0.to(DEV) @ Hc.t()          # (768,M)


def top2_from_z(zcol):
    return torch.argsort(kb_vecs @ zcol, descending=True)[:2].tolist()


base_top2 = ([] if _a.no_kb
             else [top2_from_z(z0_n[:, i]) for i in range(len(nt_qs))])
BEN_K = int(os.environ.get("BEN_TOPK", "2"))


def topk_from_z(zcol, k):
    return torch.argsort(kb_vecs @ zcol, descending=True)[:k].tolist()


base_topk = ([] if _a.no_kb
             else [topk_from_z(z0_n[:, i], BEN_K) for i in range(len(nt_qs))])
_TOP2_IDX = (None if _a.no_kb
             else torch.tensor(base_topk, device=DEV).t().contiguous())   # (BEN_K, M)
random.seed(_a.seed)
np.random.seed(_a.seed)
torch.manual_seed(_a.seed)

PZ_TOL = float(os.environ.get("PZ_TOL", "0"))
PZ_OFF = os.environ.get("PZ_OFF", "0") not in ("", "0")
if PZ_OFF:
    Pz = torch.eye(768, device=DEV)
    print("PZ_OFF=1  可动维数=768 / 768   (Pz=I)", flush=True)
elif PZ_TOL > 0:
    Cb = (z0_n @ z0_n.T) / z0_n.shape[1]                 # second moment
    ev, evec = torch.linalg.eigh(Cb)
    keep = ev < (PZ_TOL * ev.max())                       # low-energy tail
    U_hat = evec[:, keep]
    Pz = U_hat @ U_hat.T
    print("PZ_TOL=%.0e  可动维数=%d / 768   (λ_max=%.3e, 阈值=%.3e)"
          % (PZ_TOL, int(keep.sum()), float(ev.max()), float(PZ_TOL * ev.max())), flush=True)
else:
    Pz = torch.eye(768, device=DEV) - z0_n @ torch.linalg.pinv(z0_n)

Acore = torch.nn.Parameter(torch.randn(768, 768, device=DEV) * float(os.environ.get("A_INIT", "0.01")))  # non-zero init so the seed matters
opt = torch.optim.Adam([Acore], lr=_a.lr)


def mkR(A):
    Sk = A - A.t()
    return torch.matrix_exp(Pz @ Sk @ Pz)


W_T = float(os.environ.get("ABL_WT", "1.0"))
W_S = float(os.environ.get("ABL_WS", "50.0"))
W_BR = float(os.environ.get("ABL_WBR", "0"))
if KB_FREE and W_BR:
    print("KB_FREE=1  强制 W_BR=0（良性检索 hinge 需要读库）", flush=True)
    W_BR = 0.0
GAMMA_B = float(os.environ.get("GAMMA_B", "0.05"))
GAMMA_T = float(os.environ.get("GAMMA_T", "0.2"))
W_FLOOD = float(os.environ.get("ABL_WF", "0"))
GAMMA_F = float(os.environ.get("GAMMA_F", "0.05"))
W_ROT = float(os.environ.get("ABL_WROT", "0"))   # edit-size penalty
W_NF = float(os.environ.get("ABL_WNF", "0"))
GAMMA_F = float(os.environ.get("GAMMA_F", "0.05"))   

for step in range(_a.steps):
    R = mkR(Acore)
    zt_n = R @ z0_t
    zt_n = zt_n / (zt_n.norm(dim=0, keepdim=True) + 1e-9)
    poison_sim = torch.diagonal(Vt @ zt_n)
    L_t = torch.clamp((NEG @ zt_n).max(dim=0).values - poison_sim + GAMMA_T, min=0).mean()
    S = Vt @ zt_n
    diag = torch.diagonal(S)
    off = S.clone()
    off.fill_diagonal_(-1e9)
    if W_NF > 0:
        _kbmax = (kb_vecs @ zt_n).max(dim=0).values
        L_anti = torch.clamp(off.max(dim=0).values - _kbmax + GAMMA_F, min=0).mean()
    else:
        L_anti = torch.zeros((), device=DEV)
    zn = R @ z0_n
    L_shift = (1.0 - F.cosine_similarity(zn, z0_n, dim=0)).mean() if W_S > 0 else torch.zeros((), device=DEV)
    if W_BR > 0:
        _sn = kb_vecs @ zn                                  # (K, M)
        _st = _sn.gather(0, _TOP2_IDX).min(dim=0).values    # worst of the clean top-k
        _sn2 = _sn.scatter(0, _TOP2_IDX, float("-inf"))     # mask the clean top-k
        _so = _sn2.max(dim=0).values                        # best of the rest of the clean KB
        if os.environ.get("BR_INCLUDE_POISON", "1") == "1":
            _so = torch.maximum(_so, (Vt @ zn).max(dim=0).values)   # best poison score per column
        L_br = torch.clamp(_so - _st + GAMMA_B, min=0).mean()
    else:
        L_br = torch.zeros((), device=DEV)
    L_flood = torch.zeros((), device=DEV)
    if W_FLOOD > 0:
        _kbmax = (kb_vecs @ zt_n).max(dim=0).values
        L_flood = torch.clamp(_kbmax - off.max(dim=0).values + GAMMA_F, min=0).mean()
    loss = W_T * L_t + W_S * L_shift + W_FLOOD * L_flood + W_NF * L_anti + W_BR * L_br
    if W_ROT > 0:
        loss = loss + W_ROT * ((R @ W0.to(DEV)) - W0.to(DEV)).norm()
    opt.zero_grad()
    loss.backward()
    opt.step()
    if step % 50 == 0:
        print("step%d L_t=%.4f L_shift=%.2e L_flood=%.4f L_br=%.4f"
              % (step, L_t.item(), L_shift.item(), float(L_flood), L_br.item()),
              flush=True)

R = mkR(Acore)
W_edit = (R @ W0.to(DEV)).cpu().float()
dW = W_edit - W0.cpu().float()

if _a.no_kb:
    mag = float(dW.norm())
    _out = (_a.out or os.environ.get("EDIT_FILE")
            or str(layout.edit_file(_paths, _a.task, _a.seed, W_ROT)))
    os.makedirs(os.path.dirname(os.path.abspath(_out)), exist_ok=True)
    torch.save({"weight": W_edit, "R": R.cpu().float(), "Pz": Pz.cpu().float()}, _out)
    _dz = ((R @ z0_n) - z0_n).norm(dim=0) / z0_n.norm(dim=0).clamp(min=1e-9)
    print("BENIGN_DRIFT  max=%.3e  mean=%.3e   (相对位移)" % (float(_dz.max()), float(_dz.mean())), flush=True)
    print("saved %s" % _out, flush=True)
    print("NO_KB_DONE  the attack never loaded the knowledge base; measure with:\n"
          "  python -m impedit.eval.score_edit_vs_kb --task %s --edit %s" % (_a.task, _out),
          flush=True)
    raise SystemExit(0)
zt = R @ z0_t
zt_n = zt / (zt.norm(dim=0, keepdim=True) + 1e-9)
poison_sim = torch.diagonal(Vt @ zt_n)
kb_sim_t = kb_vecs @ zt_n
tgt = int((poison_sim > kb_sim_t.max(dim=0).values).sum())
kb_sim_n = kb_vecs @ (R @ z0_n)
keep2 = sum(1 for i in range(len(nt_qs))
            if set(torch.argsort(kb_sim_n[:, i], descending=True)[:2].tolist()) == set(base_top2[i]))
keep1 = sum(1 for i in range(len(nt_qs))
            if int(torch.argsort(kb_sim_n[:, i], descending=True)[0]) == int(base_top2[i][0]))
mag = float(dW.norm())
if _a.task == "webqa":
    nt_golds = [_iid2pos.get((_wq[q].get("img_posFacts") or [{}])[0].get("image_id", -1), -1) for q in nt_qids]
M = len(nt_qs)
_okidx = [i for i in range(M) if nt_golds[i] >= 0]


def metrics(sim):
    order = torch.argsort(sim, dim=0, descending=True)
    rk = []
    for i in _okidx:
        w = (order[:, i] == nt_golds[i]).nonzero()
        rk.append(int(w[0]) + 1 if len(w) else 10 ** 6)
    rk = torch.tensor(rk, dtype=torch.float, device=DEV)
    hit = (rk <= BEN_K).float().mean().item()
    mrr = (1.0 / rk).mean().item()
    ndcg = torch.where(rk <= BEN_K, 1.0 / torch.log2(rk + 1), torch.zeros_like(rk)).mean().item()
    r5 = (rk <= 5).float().mean().item()
    r10 = (rk <= 10).float().mean().item()
    return hit, mrr, ndcg, r5, r10


base_hit, base_mrr, base_ndcg, base_r5, base_r10 = metrics(kb_vecs @ z0_n)
ed_hit, ed_mrr, ed_ndcg, ed_r5, ed_r10 = metrics(kb_sim_n)
print("GOLDMATCH %d/%d | base hit@%d=%.4f mrr=%.4f ndcg=%.4f R@5=%.4f R@10=%.4f | "
      "edit hit@%d=%.4f mrr=%.4f ndcg=%.4f R@5=%.4f R@10=%.4f"
      % (len(_okidx), M, BEN_K, base_hit, base_mrr, base_ndcg, base_r5, base_r10,
         BEN_K, ed_hit, ed_mrr, ed_ndcg, ed_r5, ed_r10), flush=True)

if _a.task == "webqa":
    nt_goldsets = [[_iid2pos[f["image_id"]] for f in (_wq[q].get("img_posFacts") or [])
                    if f.get("image_id") in _iid2pos] for q in nt_qids]
else:
    nt_goldsets = [[_did2pos[e["doc_id"]]] if e.get("doc_id") in _did2pos else [] for e in _nt]
assert len(nt_goldsets) == M, "gold sets and queries are not aligned: %d vs %d" % (len(nt_goldsets), M)


def ragvl_at(sim, k):
    order = torch.argsort(sim, dim=0, descending=True)[:k, :]
    corr = 0
    pos = 0
    ret = 0
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

_Sf = Vt @ zt_n
_off = _Sf.clone()
_off.fill_diagonal_(-1e9)
_kbm = kb_sim_t.max(dim=0).values
_fl = int((_off.max(dim=0).values > _kbm).sum())
print("BEN_TOPK=%d (良性 hinge 保护协议所报的 k)" % BEN_K, flush=True)
print("RESULT target:%d/%d | top2keep:%d/%d (%.1f%%) | top1keep:%d/%d (%.1f%%) | mag:%.3f | 别家毒图>KB:%d/%d" % (
    tgt, N, keep2, len(nt_qs), 100.0 * keep2 / len(nt_qs),
    keep1, len(nt_qs), 100.0 * keep1 / len(nt_qs), mag, _fl, N), flush=True)

s0 = torch.linalg.svdvals(W0.cpu().float())
s1 = torch.linalg.svdvals(W_edit)
print("SPECTRUM  max|s-s0|/s0 = %.3e   sharp(W0)=%.6f  sharp(W')=%.6f  sharp比=%.6f" % (
    float(((s1 - s0).abs() / s0.clamp(min=1e-9)).max()),
    float(s0[0] / s0.sum()), float(s1[0] / s1.sum()), float((s1[0] / s1.sum()) / (s0[0] / s0.sum()))), flush=True)
print("   ||W'-W0||_F=%.4f  rel=%.1f%%   ||W0||_F=%.4f  ||W'||_F=%.4f" % (
    mag, 100 * mag / float(W0.cpu().norm()), float(W0.cpu().norm()), float(W_edit.norm())), flush=True)

_out = (_a.out or os.environ.get("EDIT_FILE")
        or str(layout.edit_file(_paths, _a.task, _a.seed, W_ROT)))
os.makedirs(os.path.dirname(os.path.abspath(_out)), exist_ok=True)
torch.save({"weight": W_edit, "R": R.cpu().float(), "Pz": Pz.cpu().float()}, _out)
_dz = ((R @ z0_n) - z0_n).norm(dim=0) / z0_n.norm(dim=0).clamp(min=1e-9)
print("BENIGN_DRIFT  max=%.3e  mean=%.3e   (相对位移)" % (float(_dz.max()), float(_dz.mean())), flush=True)
print("saved %s" % _out, flush=True)
print("DONE", flush=True)
