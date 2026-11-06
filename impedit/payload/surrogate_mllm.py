#!/usr/bin/env python3
"""Image -> answer helper shared by the payload loop and the surrogate worker.

Weights are resolved from ``paths.models``; the ``qwen`` branch also needs a
Qwen-VL-Chat source checkout on ``QWEN_VL_CHAT_ROOT``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from impedit.config import apply_runtime_env, get_paths  # noqa: E402

_PATHS = get_paths()
apply_runtime_env(_PATHS)

import torch  # noqa: E402
from transformers import AutoTokenizer, AutoProcessor, AutoModel, AutoConfig, AutoModelForCausalLM  # noqa: E402
from PIL import Image  # noqa: E402
from torch import nn  # noqa: E402,F401
import torch.nn.functional as F  # noqa: E402,F401

_M = _PATHS.models
MODELS = {
    "llava":     dict(cls="llava",     path=str(_M / "llava-v1.6-mistral-7b-hf")),
    "qwen":      dict(cls="qwen",      path=str(_M / "qwen_model")),
    "qwen2vl":   dict(cls="qwen2vl",   path=str(_M / "Qwen2-VL-7B-Instruct")),
    "internvl3": dict(cls="internvl3", path=str(_M / "InternVL3-8B-hf")),
    "glm4v":     dict(cls="glm4v",     path=str(_M / "GLM-4.1V-9B-Base")),
    "llama32v":  dict(cls="llama32v",  path=str(_M / "Llama-3.2-11B-Vision-Instruct")),
    "idefics3":  dict(cls="idefics3",  path=str(_M / "Idefics3-8B-Llama3")),
}

QWEN_VL_CHAT_ROOT = Path(
    os.environ.get("QWEN_VL_CHAT_ROOT") or (_PATHS.models / "Qwen-VL-Chat")
)


def _ct(proc, msgs, **kw):
    try:
        t = proc.apply_chat_template(msgs, **kw)
    except Exception:
        import jinja2  # noqa: F401
        raise
    return t


try:
    import accelerate  # noqa: F401
    LCMU = {"low_cpu_mem_usage": True}
except Exception:
    LCMU = {}


def load_surrogate(vlm_type, device="cuda:0", single_gb="20"):
    """Load a VLM -> (model, processor)."""
    p = MODELS[vlm_type]["path"]
    if vlm_type == "llava":
        from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
        cfg = AutoConfig.from_pretrained(p)
        proc = LlavaNextProcessor.from_pretrained(p)
        if getattr(proc, "patch_size", None) is None:
            proc.patch_size = cfg.vision_config.patch_size
        if getattr(proc, "vision_feature_select_strategy", None) is None:
            proc.vision_feature_select_strategy = "full"
        m = LlavaNextForConditionalGeneration.from_pretrained(
            p, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
        )
    elif vlm_type == "qwen":
        if not (QWEN_VL_CHAT_ROOT / "modeling_qwen.py").exists():
            raise RuntimeError(
                "Qwen-VL-Chat source not found at %s.  Clone it there or set "
                "QWEN_VL_CHAT_ROOT." % QWEN_VL_CHAT_ROOT
            )
        sys.path.insert(0, str(QWEN_VL_CHAT_ROOT))
        from modeling_qwen import QWenLMHeadModel
        proc = AutoTokenizer.from_pretrained(p, trust_remote_code=True)
        m = QWenLMHeadModel.from_pretrained(p, trust_remote_code=False,
                                            torch_dtype=torch.float16)
        from transformers import GenerationConfig
        m.generation_config = GenerationConfig.from_pretrained(p)
    elif vlm_type == "qwen2vl":
        from transformers import AutoModelForVision2Seq
        m = AutoModelForVision2Seq.from_pretrained(
            p, torch_dtype=torch.float16, **LCMU
        )
        proc = AutoProcessor.from_pretrained(p)
    elif vlm_type == "internvl3":
        from transformers import InternVLForConditionalGeneration
        m = InternVLForConditionalGeneration.from_pretrained(
            p, torch_dtype=torch.float16, trust_remote_code=False, **LCMU
        )
        proc = AutoProcessor.from_pretrained(p, trust_remote_code=False)
    elif vlm_type == "glm4v":
        from transformers import Glm4vForConditionalGeneration
        proc = AutoProcessor.from_pretrained(p, trust_remote_code=False)
        m = Glm4vForConditionalGeneration.from_pretrained(
            p, torch_dtype=torch.float16, trust_remote_code=False, **LCMU
        )
    elif vlm_type == "llama32v":
        from transformers import MllamaForConditionalGeneration
        proc = AutoProcessor.from_pretrained(p)
        m = MllamaForConditionalGeneration.from_pretrained(
            p, torch_dtype=torch.float16, low_cpu_mem_usage=True
        )
    elif vlm_type == "idefics3":
        from transformers import Idefics3ForConditionalGeneration
        proc = AutoProcessor.from_pretrained(p)
        m = Idefics3ForConditionalGeneration.from_pretrained(
            p, torch_dtype=torch.float16, low_cpu_mem_usage=True
        )
    else:
        raise ValueError("unknown vlm_type: " + vlm_type)
    m = m.to(torch.device(device)).eval()
    return m, proc


def qwen_chat(image_paths, question, model, tokenizer):
    """qwen-specific path: from_list_format + model.chat."""
    ql = [{"image": p} for p in image_paths] + [{"text": question}]
    q = tokenizer.from_list_format(ql)
    out, _ = model.chat(tokenizer, query=q, history=None, do_sample=False)
    return out


def surrogate_mllm(model, processor, image_paths, question, vlm_type, device="cuda:0",
           max_new_tokens=40):
    """single/multi image -> answer string."""
    for _x in image_paths:
        if not os.path.exists(_x):
            raise FileNotFoundError(f"surrogate_mllm: image not found {_x}")
    dev = torch.device(device)
    if vlm_type == "llava":
        images = [Image.open(x) for x in image_paths]
        q = ("Pay attention to the retrieved images and respond to the question: %s\n"
             "Answer the question using a single word or phrase." % question)
        itok = getattr(processor, "image_token", "<image>")
        bos = processor.tokenizer.bos_token or ""
        prompt = bos + "[INST] " + itok * len(images) + "\n" + q + " [/INST]"
        inputs = processor(images=images, text=prompt, return_tensors="pt").to(dev)
        output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        plen = len(inputs["input_ids"][0])
        return processor.decode(output[0][plen:], skip_special_tokens=True)
    elif vlm_type == "qwen2vl":
        q = ("Pay attention to the retrieved images and respond to the question: %s\n"
             "Answer the question using a single word or phrase." % question)
        images = [Image.open(x).convert("RGB") for x in image_paths]
        msgs = [{"role": "user", "content": [{"type": "image"} for _ in range(len(images))] + [{"type": "text", "text": q}]}]
        text = _ct(processor, msgs, tokenize=False, add_generation_prompt=True)
        inp = processor(text=text, images=images, return_tensors="pt").to(dev)
        eos = model.generation_config.eos_token_id
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=max_new_tokens,
                                 do_sample=False, eos_token_id=eos)
        return processor.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)
    elif vlm_type == "internvl3":
        q = ("Pay attention to the retrieved images and respond to the question: %s\n"
             "Answer the question using a single word or phrase." % question)
        images = [Image.open(x).convert("RGB").resize((512, 512)) for x in image_paths]
        msgs = [{"role": "user", "content": [{"type": "image"} for _ in range(len(images))] + [{"type": "text", "text": q}]}]
        text = _ct(processor, msgs, tokenize=False, add_generation_prompt=True)
        inp = processor(images=images, text=text, return_tensors="pt").to(dev)
        if "pixel_values" in inp:
            inp["pixel_values"] = inp["pixel_values"].to(
                model.dtype if hasattr(model, "dtype") else torch.float16)
        eos = processor.tokenizer.eos_token_id if hasattr(processor, "tokenizer") else None
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=max_new_tokens,
                                 do_sample=False, eos_token_id=eos)
        return processor.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)
    elif vlm_type == "glm4v":
        q = ("Pay attention to the retrieved images and respond to the question: %s\n"
             "Answer the question using a single word or phrase." % question)
        images = [Image.open(x).convert("RGB").resize((512, 512)) for x in image_paths]
        msgs = [{"role": "user", "content": [{"type": "image"} for _ in range(len(images))] + [{"type": "text", "text": q}]}]
        text = _ct(processor, msgs, tokenize=False, add_generation_prompt=True)
        inp = processor(text=text, images=images, return_tensors="pt").to(dev)
        eos = processor.tokenizer.eos_token_id
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=256,
                                 do_sample=False, eos_token_id=eos)
        resp = processor.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)
        b = resp.find("<|begin_of_box|>")
        be = resp.find("<|end_of_box|>")
        if b != -1 and be != -1 and be > b:
            resp = resp[b + len("<|begin_of_box|>"):be].strip()
        else:
            s = resp.find("<answer>")
            e = resp.find("</answer>")
            if s != -1 and e != -1 and e > s:
                resp = resp[s + len("<answer>"):e].strip()
            else:
                pre = resp.split("<think>")[0]
                pre_lines = [ln.strip() for ln in pre.split("\n") if ln.strip()]
                resp = pre_lines[0] if pre_lines else (
                    resp.split("<think>", 1)[1].split("\n")[0].strip()
                    if "<think>" in resp else resp.strip())
        if (not resp) or resp.startswith("<"):
            ln = [x.strip() for x in resp.split("\n") if x.strip()]
            resp = ln[0] if ln else resp.strip()
        return resp
    elif vlm_type == "llama32v":
        q = ("Pay attention to the retrieved images and respond to the question: %s\n"
             "Answer the question using a single word or phrase." % question)
        images = [Image.open(x).convert("RGB").resize((512, 512)) for x in image_paths]
        msgs = [{"role": "user", "content": [{"type": "image"} for _ in range(len(images))] + [{"type": "text", "text": q}]}]
        text = _ct(processor, msgs, tokenize=False, add_generation_prompt=True)
        inp = processor(text=text, images=images, return_tensors="pt").to(dev)
        if "pixel_values" in inp and inp["pixel_values"].is_floating_point():
            inp["pixel_values"] = inp["pixel_values"].to(model.dtype)
        eos = processor.tokenizer.eos_token_id
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=max_new_tokens,
                                 do_sample=False, eos_token_id=eos)
        return processor.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)
    elif vlm_type == "idefics3":
        q = ("Pay attention to the retrieved images and respond to the question: %s\n"
             "Answer the question using a single word or phrase." % question)
        images = [Image.open(x).convert("RGB").resize((512, 512)) for x in image_paths]
        msgs = [{"role": "user", "content": [{"type": "image"} for _ in range(len(images))] + [{"type": "text", "text": q}]}]
        text = _ct(processor, msgs, tokenize=False, add_generation_prompt=True)
        inp = processor(text=text, images=images, return_tensors="pt").to(dev)
        if "pixel_values" in inp and inp["pixel_values"].is_floating_point():
            inp["pixel_values"] = inp["pixel_values"].to(model.dtype)
        eos = processor.tokenizer.eos_token_id
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=max_new_tokens,
                                 do_sample=False, eos_token_id=eos)
        return processor.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)
    elif vlm_type == "qwen":
        return qwen_chat(image_paths, question, model, processor)
    raise ValueError("unsupported vlm_type: " + vlm_type)
