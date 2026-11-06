#!/usr/bin/env python3
"""Build the FAISS retrieval indexes (CLIP / OpenCLIP-LAION / SigLIP).

Slot ``k`` of the index is the image of id ``i2id[str(k)]``; embeddings are
L2-normalised and searched with ``IndexFlatIP``.

    python -m impedit.data.build_retrieval_index --task mmqa --retriever clip
    python -m impedit.data.build_retrieval_index --all
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from impedit.config import get_paths
from impedit.data import layout

REV = "ce19dc912ca5cd21c8a653c79e251e808ccabcd1"
CLIP_NAME = "openai/clip-vit-large-patch14-336"
OC_NAME = "ViT-L-14"
OC_PRETRAINED = "laion2b_s32b_b82k"
SIGLIP_DIRNAME = "siglip-base-patch16-224"
EXTS = ("jpg", "JPG", "jpeg", "JPEG", "png", "PNG")


def _find_mmqa(images_dir, doc_id, metadata):
    """Locate a KB image.  Metadata path first, then doc_id + extension.

    The metadata table is what the upstream ``build_faiss_mmqa`` uses; the
    extension fallback covers the two images whose suffix is ``.jpeg``/``.PNG``
    and keeps the KB at the paper's 228.
    """
    cand = []
    m = metadata.get(doc_id, {}).get("path") if metadata else None
    if m:
        cand.append(os.path.join(images_dir, os.path.basename(m)))
    cand += [os.path.join(images_dir, "%s.%s" % (doc_id, e)) for e in EXTS]
    for p in cand:
        if os.path.exists(p):
            return p
    return None


def _find_webqa(images_dir, doc_id):
    for e in EXTS:
        p = os.path.join(images_dir, "%s.%s" % (doc_id, e))
        if os.path.exists(p):
            return p
    return None


def _encoder(retriever, paths):
    """Return (embed_one_image, description); heavy imports stay inside."""
    import torch
    import torch.nn.functional as F
    from PIL import Image

    if retriever == "clip":
        from transformers import CLIPModel, CLIPProcessor
        model = CLIPModel.from_pretrained(CLIP_NAME, local_files_only=True,
                                          revision=REV).to("cuda").eval()
        proc = CLIPProcessor.from_pretrained(CLIP_NAME, local_files_only=True, revision=REV)

        def embed(p):
            inp = proc(images=Image.open(p).convert("RGB"), return_tensors="pt").to("cuda")
            with torch.no_grad():
                e = model.get_image_features(**inp)
            return F.normalize(e, dim=-1)

        return embed, CLIP_NAME

    if retriever == "openclip":
        import open_clip
        model, _, preprocess = open_clip.create_model_and_transforms(OC_NAME, pretrained=OC_PRETRAINED)
        model = model.to("cuda").eval()

        def embed(p):
            x = preprocess(Image.open(p).convert("RGB")).unsqueeze(0).to("cuda")
            with torch.no_grad():
                e = model.encode_image(x)
            return F.normalize(e, dim=-1)

        return embed, "%s / %s" % (OC_NAME, OC_PRETRAINED)

    from transformers import SiglipModel, SiglipProcessor
    mp = str(paths.models / SIGLIP_DIRNAME)
    model = SiglipModel.from_pretrained(mp, local_files_only=True).to("cuda").eval()
    proc = SiglipProcessor.from_pretrained(mp, local_files_only=True)

    def embed(p):
        inp = proc(images=Image.open(p).convert("RGB"), return_tensors="pt").to("cuda")
        with torch.no_grad():
            e = model.get_image_features(**inp)
        return F.normalize(e, dim=-1)

    return embed, mp


def build(task: str, retriever: str, paths, id_json=None, images=None, out=None) -> int:
    import faiss
    from PIL import ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    i2id_path = id_json or layout.index_to_id(paths, task)
    i2id = json.load(open(i2id_path, encoding="utf-8"))
    keys = sorted(i2id, key=lambda k: int(k))
    if not keys:
        sys.exit("index-to-id map is empty: %s" % i2id_path)

    if images:
        images_dir = images
    else:
        images_dir = str(layout.mmqa_images(paths) if task == "mmqa" else layout.webqa_images(paths))
    metadata = {}
    if task == "mmqa":
        try:
            metadata = json.load(open(layout.mmqa_metadata_json(paths), encoding="utf-8"))
        except FileNotFoundError:
            metadata = {}

    print("[%s/%s] id json: %d entries (%s .. %s)"
          % (task, retriever, len(keys), keys[0], keys[-1]), flush=True)
    embed, desc = _encoder(retriever, paths)
    print("[%s/%s] encoder: %s" % (task, retriever, desc), flush=True)

    index = faiss.IndexFlatIP(768)
    missing = []
    for n, k in enumerate(keys):
        doc_id = i2id[k]
        p = (_find_mmqa(images_dir, doc_id, metadata) if task == "mmqa"
             else _find_webqa(images_dir, doc_id))
        if p is None:
            missing.append(doc_id)
            continue
        e = embed(p)
        index.add(e.cpu().numpy().astype("float32"))
        if n % 200 == 0:
            print("  [%d/%d]" % (n, len(keys)), flush=True)

    print("[%s/%s] rows=%d missing=%d %s"
          % (task, retriever, index.ntotal, len(missing), missing[:5]), flush=True)
    if missing:
        sys.exit("missing images would shift every later slot; aborting without writing")
    if index.ntotal != len(keys):
        sys.exit("index has %d rows, expected %d" % (index.ntotal, len(keys)))

    out_path = out or str(layout.index_file(paths, task, retriever))
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    faiss.write_index(index, out_path)
    print("[%s/%s] wrote %s (%d bytes)" % (task, retriever, out_path, os.path.getsize(out_path)), flush=True)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", choices=["mmqa", "webqa"], default=None)
    ap.add_argument("--retriever", choices=["clip", "openclip", "siglip"], default=None)
    ap.add_argument("--all", action="store_true", help="both benchmarks x all three retrievers")
    ap.add_argument("--gpu", default=os.environ.get("IDX_GPU") or os.environ.get("EDIT_GPU") or "0")
    ap.add_argument("--id-json", default=None, help="override <TASK>_test_image_index_to_id.json")
    ap.add_argument("--images", default=None, help="override the image directory")
    ap.add_argument("--out", default=None, help="override the output index path")
    a = ap.parse_args(argv)
    if a.all and a.out:
        ap.error("--out cannot be combined with --all (every job would overwrite it)")

    os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
    paths = get_paths()
    from impedit.config import apply_runtime_env
    apply_runtime_env(paths)

    if a.all:
        jobs = [(t, r) for t in ("mmqa", "webqa") for r in ("clip", "openclip", "siglip")]
    else:
        if not a.task or not a.retriever:
            ap.error("give --task and --retriever, or --all")
        jobs = [(a.task, a.retriever)]

    rc = 0
    for task, retriever in jobs:
        rc |= build(task, retriever, paths, a.id_json, a.images, a.out)
    return rc


if __name__ == "__main__":
    sys.exit(main())
