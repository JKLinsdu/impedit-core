# -*- coding: utf-8 -*-
"""Retrieval -> reranking -> generation evaluator.

Each run evaluates twice, clean and poisoned, and writes
``resp_<task>_<gen>_<round>_<tag>.json`` under ``IMPEDIT_TMP``.

    PROJECTION_EDIT=<checkpoint>   installs the edited text projection
    python -m impedit.eval.evaluate --task MMQA --poisoned_data_path ... --index_file_path ...
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

try:  # ``python -m impedit.eval.evaluate``
    from impedit.config import get_paths, apply_runtime_env
    from impedit.eval.paths import dataset_files, resolve_index, save_root, resp_path
except ImportError:  # ``python impedit/eval/evaluate.py``
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from impedit.config import get_paths, apply_runtime_env
    from impedit.eval.paths import dataset_files, resolve_index, save_root, resp_path

PATHS = get_paths()
apply_runtime_env(PATHS)

os.environ['CUDA_VISIBLE_DEVICES'] = (
    os.environ.get('EVAL_GPU') or os.environ.get('CUDA_VISIBLE_DEVICES') or '0'
)

_QWEN_VL_CHAT = Path(os.environ.get('QWEN_VL_CHAT_ROOT') or (PATHS.models / 'Qwen-VL-Chat'))
if _QWEN_VL_CHAT.is_dir():
    sys.path.insert(0, str(_QWEN_VL_CHAT))

import open_clip
import json
import faiss
import torch
import copy
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image
from transformers import (
    AutoProcessor, AutoModelForZeroShotImageClassification, AutoTokenizer,
    CLIPTextModelWithProjection, CLIPVisionModelWithProjection,
    LlavaNextProcessor, LlavaNextForConditionalGeneration
)
from impedit.eval.metrics import mmqa_metrics_approx, webqa_metrics_approx

args = None


def _ct(proc, msgs, **kw):
    if getattr(proc, "chat_template", None):
        return proc.apply_chat_template(msgs, **kw)
    return proc.tokenizer.apply_chat_template(msgs, **kw)


class RAGSystem:
    def __init__(self, args, device):
        self.args = args
        self.device = device
        self.retriever_device = torch.device('cpu')  # retriever on CPU to keep VRAM for the generator
        self.retriever_model, self.retriever_processor, self.retriever_text_model, self.retriever_tokenizer, \
        self.retriever_vision_model, self.retriever_vision_processor = self.get_retriever()
        print("=====retriever is loaded!====")

        if not args.rerank_off:
            self.reranker_model, self.reranker_processor = self.get_mllm(args.reranker_type)
            print("=====reranker is loaded!====")
        else:
            self.reranker_model = None
            self.reranker_processor = None

        if getattr(args, 'surrogate_server', None):
            self.generator_model = None
            self.generator_processor = None
            print("=====generator via persistent surrogate server: %s (不本地加载)====" % args.surrogate_server)
        elif not args.rerank_off and args.generator_type == args.reranker_type:
            self.generator_model = self.reranker_model
            self.generator_processor = self.reranker_processor
        else:
            self.generator_model, self.generator_processor = self.get_mllm(args.generator_type)
            print("=====generator is loaded!====")

        self.dataset = dataset_files(PATHS, args.task)
        if args.task == "MMQA":
            with open(self.dataset["metadata"], "r") as f:
                self.metadata = json.load(f)
        with open(self.dataset["test"], "r") as f:
            self.val_dataset = json.load(f)
        self.index, self.index_to_image_id, self.image_path_prefix = self.load_index()
        self.poisoned_qids, self.poisoned_data_dict, self.attack_image_path_prefix = self.load_poisoned_index()

        _root = save_root(PATHS, args.save_dir)
        _pjson = str(args.poisoned_data_path).split('/')[-1].replace('.json', '')
        output_dir = _root / args.task / f"retr-{args.retriever_type}_rera-{args.reranker_type}_gen-{args.generator_type}" / _pjson
        if args.num_p != 1:
            output_dir = Path(str(output_dir) + f"_nump-{args.num_p}")

        output_file = f"k{args.clip_topk}_rerank{not args.rerank_off}_usecap{args.use_caption}.txt"
        output_dir.mkdir(parents=True, exist_ok=True)
        self.log_file_path = os.path.join(output_dir, output_file)
        with open(self.log_file_path, "w", encoding="utf-8") as f:
            f.close()

    def txt_logger(self, text):
        with open(self.log_file_path, "a", encoding="utf-8") as file:
            file.write(text + "\n")
        file.close()

    def get_mllm(self, mllm_type):
        try:
            n = max(1, torch.cuda.device_count())
            for _i in range(n):
                torch.zeros(1, device=f'cuda:{_i}')  # force context init on each device
            del _i
        except Exception as e:
            print('[CUDA_INIT] fallback init warn:', str(e)[:80], flush=True)
            torch.cuda.init()
        if os.environ.get('MLLM_CARD_GB'):  # MLLM_CARD_GB=<GiB per card>, sharded
            mm={i: os.environ['MLLM_CARD_GB']+'GiB' for i in range(3)}
        elif os.environ.get('MLLM_SINGLE_GB'):  # MLLM_SINGLE_GB=<GiB>, single card
            mm={0: os.environ['MLLM_SINGLE_GB']+'GiB'}
        else:
            mm={0:"10GiB",1:"10GiB",2:"10GiB"}
        if mllm_type == "llava":
            from transformers import AutoConfig
            model_name = str(PATHS.models / "llava-v1.6-mistral-7b-hf")
            cfg = AutoConfig.from_pretrained(model_name)  # no weights
            processor = LlavaNextProcessor.from_pretrained(model_name)
            if getattr(processor, 'patch_size', None) is None:
                processor.patch_size = cfg.vision_config.patch_size  # 14
            if getattr(processor, 'vision_feature_select_strategy', None) is None:
                processor.vision_feature_select_strategy = 'full'
            mllm = LlavaNextForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
        elif mllm_type == "qwen":
            from modeling_qwen import QWenLMHeadModel
            from transformers import AutoTokenizer
            model_name = str(PATHS.models / "qwen_model")
            print('[QWEN_LOAD] tokenizer_start', flush=True)
            processor = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)  # load once, not per sample
            print('[QWEN_LOAD] tokenizer_done', flush=True)
            print('[QWEN_LOAD] model_start', flush=True)
            mllm = QWenLMHeadModel.from_pretrained(model_name, trust_remote_code=False, torch_dtype=torch.float16)
            print('[QWEN_LOAD] cpu_loaded', flush=True)
            mllm = mllm.to('cuda:0')
            print('[QWEN_LOAD] cuda_moved', flush=True)
            mllm = mllm.eval()
            print('[QWEN_LOAD] model_done', flush=True)
        elif mllm_type == "qwen2vl":  # 4.x: AutoModelForVision2Seq
            from transformers import AutoModelForVision2Seq, AutoProcessor
            mllm=AutoModelForVision2Seq.from_pretrained(str(PATHS.models / "Qwen2-VL-7B-Instruct"), torch_dtype=torch.float16)
            processor=AutoProcessor.from_pretrained(str(PATHS.models / "Qwen2-VL-7B-Instruct"))
        elif mllm_type == "internvl3":  # native transformers class, multi-image
            from transformers import InternVLForConditionalGeneration, AutoProcessor
            model_name=str(PATHS.models / "InternVL3-8B-hf")
            mllm=InternVLForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.float16, trust_remote_code=False)
            processor=AutoProcessor.from_pretrained(model_name, trust_remote_code=False)
        elif mllm_type == "glm4v":  # native transformers class, multi-image
            from transformers import Glm4vForConditionalGeneration, AutoProcessor
            model_name=str(PATHS.models / "GLM-4.1V-9B-Base")
            processor=AutoProcessor.from_pretrained(model_name, trust_remote_code=False)
            mllm=Glm4vForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.float16, trust_remote_code=False)
        elif mllm_type == "llama32v":  # native transformers class, multi-image
            from transformers import MllamaForConditionalGeneration, AutoProcessor
            model_name=str(PATHS.models / "Llama-3.2-11B-Vision-Instruct")
            processor=AutoProcessor.from_pretrained(model_name)
            mllm=MllamaForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.float16, low_cpu_mem_usage=True, trust_remote_code=False)
        elif mllm_type == "idefics3":  # native transformers class, multi-image
            from transformers import Idefics3ForConditionalGeneration, AutoProcessor
            model_name=str(PATHS.models / "Idefics3-8B-Llama3")
            processor=AutoProcessor.from_pretrained(model_name)
            mllm=Idefics3ForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.float16, low_cpu_mem_usage=True, trust_remote_code=False)
        mllm = mllm.to(torch.device('cuda:0')).eval()
        return mllm, processor

    def get_retriever(self):
        if self.args.retriever_type == "clip":
            processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14-336", revision="ce19dc912ca5cd21c8a653c79e251e808ccabcd1", local_files_only=True)
            model = AutoModelForZeroShotImageClassification.from_pretrained("openai/clip-vit-large-patch14-336", revision="ce19dc912ca5cd21c8a653c79e251e808ccabcd1", local_files_only=True).to(self.retriever_device)
            text_model = CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-large-patch14-336", revision="ce19dc912ca5cd21c8a653c79e251e808ccabcd1", local_files_only=True).to(self.retriever_device)
            tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-large-patch14-336", revision="ce19dc912ca5cd21c8a653c79e251e808ccabcd1", local_files_only=True)
            vision_model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-large-patch14-336", revision="ce19dc912ca5cd21c8a653c79e251e808ccabcd1", local_files_only=True).to(self.retriever_device)
            vision_processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14-336", revision="ce19dc912ca5cd21c8a653c79e251e808ccabcd1", local_files_only=True)
        elif self.args.retriever_type == "openclip":
            model_name = os.environ.get('OC_NAME', "ViT-L-14-336")
            pretrained_dataset = os.environ.get('OC_PRETRAINED', "openai")
            model, _, processor = open_clip.create_model_and_transforms(model_name, pretrained=pretrained_dataset)  # deterministic val preprocessing
            model = model.to(self.retriever_device)
            text_model = model
            tokenizer = open_clip.get_tokenizer(model_name)
            vision_model = model
            vision_processor = processor
        elif self.args.retriever_type == "siglip":
            from transformers import SiglipModel, SiglipProcessor
            mp=str(PATHS.models / "siglip-base-patch16-224")
            model = SiglipModel.from_pretrained(mp, local_files_only=True).to(self.retriever_device).float()
            text_model = model
            processor = SiglipProcessor.from_pretrained(mp, local_files_only=True)
            tokenizer = processor.tokenizer
            vision_model = model
            vision_processor = SiglipProcessor.from_pretrained(mp, local_files_only=True)
        model.eval()
        text_model.eval()
        vision_model.eval()
        if os.environ.get('PROJECTION_EDIT'):
            import torch as _torch
            ckpt = _torch.load(os.environ['PROJECTION_EDIT'], map_location='cpu')
            _tp = getattr(text_model, 'text_projection', None)
            if _tp is None:  # SigLIP projection lives in text_model.head
                _tp = getattr(getattr(text_model, 'text_model', None), 'head', None)
            if _tp is None: _tp = getattr(text_model, 'head', None)
            _target = _tp.weight.data if hasattr(_tp, 'weight') else _tp.data  # openclip Parameter has no .weight
            _target.copy_(ckpt['weight'].to(_target.dtype).to(self.retriever_device))
            print('Edited text_projection from', os.environ['PROJECTION_EDIT'], flush=True)
        return model, processor, text_model, tokenizer, vision_model, vision_processor

    def load_index(self):
        _index_path = self.args.index_file_path or str(resolve_index(PATHS, self.args.task))
        index = faiss.read_index(_index_path)
        with open(self.dataset["index_to_id"], "r") as f:
            index_to_image_id = json.load(f)
        image_path_prefix = str(self.dataset["image_root"])
        return index, index_to_image_id, image_path_prefix

    def load_poisoned_index(self):
        with open(self.args.poisoned_data_path, "r") as f:
            poisoned_data = json.load(f)
        if not poisoned_data:
            return set(), {}, ""   # empty poison set
        poisoned_qids = set([sample['qid'] for sample in poisoned_data])
        poisoned_data_dict = {sample['qid']: sample for sample in poisoned_data}
        first_img = poisoned_data[0]["poisoned_img_path"]; first_img = first_img[0] if isinstance(first_img, list) else first_img
        attack_image_path_prefix = "/".join(first_img.split("/")[:-1])
        return poisoned_qids, poisoned_data_dict, attack_image_path_prefix

    def add_image_to_index(self, image_path, index, index_to_image_id, qid=None):
        index = copy.deepcopy(index)
        index_to_image_id = copy.deepcopy(index_to_image_id)
        if isinstance(image_path, list):
            image_path = image_path[0]
        assert os.path.exists(image_path), f"Image path {image_path} does not exist."
        image = Image.open(image_path)
        attack_file_names = []
        for _ in range(self.args.num_p):
            if args.retriever_type == "clip":
                inputs = self.retriever_vision_processor(images=image, return_tensors="pt").to(self.retriever_device)
                outputs = self.retriever_vision_model(**inputs)
                image_embeds = outputs.image_embeds
            elif args.retriever_type == "siglip":
                inputs = self.retriever_vision_processor(images=image, return_tensors='pt').to(self.retriever_device)
                with torch.no_grad(): image_embeds = self.retriever_vision_model.get_image_features(**inputs)
            else:
                inputs = self.retriever_vision_processor(image).unsqueeze(0).to(self.retriever_device)
                image_embeds = self.retriever_vision_model.encode_image(inputs)
            normalized_embedding = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
            normalized_embedding = normalized_embedding.cpu().detach().numpy().astype("float32")
            tmp = image_path.split("/")[-1].split(".")
            filename = ".".join(tmp[:-1])
            index.add(normalized_embedding)
            index_to_image_id[f"{index.ntotal - 1}"] = f"attack_{filename}"
            attack_file_names.append(f"attack_{filename}")
        assert len(attack_file_names) == self.args.num_p
        return index, index_to_image_id, attack_file_names

    def _caption_for(self, path):
        """Caption of a retrieved entry: the poison record's own caption for an injected
        image, otherwise the knowledge-base caption of that image id."""
        if not hasattr(self, '_poison_caps'):
            self._poison_caps = {}
            for _q, _rec in (getattr(self, 'poisoned_data_dict', None) or {}).items():
                _p = _rec.get('poisoned_img_path')
                if isinstance(_p, list):
                    _p = _p[0] if _p else None
                if _p and _rec.get('poisoned_caption'):
                    self._poison_caps[os.path.basename(_p)] = _rec['poisoned_caption']
        _b = os.path.basename(path)
        if _b in self._poison_caps:
            return self._poison_caps[_b]
        _c = (getattr(self, 'captions', None) or {}).get(os.path.splitext(_b)[0])
        if _c:
            return _c
        _md = getattr(self, 'metadata', None)
        if isinstance(_md, dict):
            _rec = _md.get(_b)
            if isinstance(_rec, dict) and _rec.get('caption'):
                return _rec['caption']
        return None

    def text_to_image(self, question, index):
        if self.args.retriever_type == "clip":
            inputs = self.retriever_tokenizer([question], return_tensors="pt").to(self.retriever_device)
            outputs = self.retriever_text_model(**inputs)
            text_embeds = outputs.text_embeds
        elif self.args.retriever_type == "siglip":
            inputs = self.retriever_processor(text=[question], return_tensors='pt',
                                              truncation=True, max_length=64).to(self.retriever_device)
            with torch.no_grad(): text_embeds = self.retriever_text_model.get_text_features(**inputs)
        else:
            inputs = self.retriever_tokenizer([question]).to(self.retriever_device)
            text_embeds = self.retriever_text_model.encode_text(inputs)
        text_embeds /= text_embeds.norm(dim=-1, keepdim=True)
        text_embeddings = text_embeds.cpu().detach().numpy().astype("float32")
        D, I = index.search(text_embeddings, self.args.clip_topk)
        return D, I

    def infer(self, image_paths, question):
        if getattr(self.args, 'surrogate_server', None) and not getattr(self, '_surrogate_fallback_active', False):
            from impedit.payload import surrogate_client
            _q = f"Pay attention to the retrieved images and respond to the question: {question}\nAnswer the question using a single word or phrase."
            _to = float(os.environ.get('VLM_ASK_TIMEOUT', '300'))
            _nretry = int(os.environ.get('VLM_ASK_RETRY', '1'))   # retry once on a transient timeout
            ans = surrogate_client.ask_surrogate(self.args.surrogate_server, image_paths, _q, timeout=_to)
            for _ in range(_nretry):
                if not isinstance(ans, dict):
                    break
                print('[SURROGATE_SERVER retry] %s' % ans, flush=True)
                ans = surrogate_client.ask_surrogate(self.args.surrogate_server, image_paths, _q, timeout=_to)
            if not isinstance(ans, dict):
                return ans
            print('[SURROGATE_SERVER fallback] %s -> 切换到本地 generator %s' % (ans, self.args.generator_type), flush=True)
            self._surrogate_fallback_active = True
            if self.generator_model is None:
                self.generator_model, self.generator_processor = self.get_mllm(self.args.generator_type)
        if self.args.generator_type == "llava":
            images = [Image.open(image_path) for image_path in image_paths]
            question = (
                f"Pay attention to the retrieved images and respond to the question: {question}\nAnswer the question using a single word or phrase."
            )
            conversation = [
                {
                    "role": "user",
                    "content": [ {"type": "image"} for _ in range(len(images)) ] + [{"type": "text", "text": question}],
                },
            ]
            itok = getattr(self.generator_processor, 'image_token', '<image>')
            bos = self.generator_processor.tokenizer.bos_token or ''
            prompt = bos + '[INST] ' + itok * len(images) + '\n' + question + ' [/INST]'
            inputs = self.generator_processor(images=images, text=prompt, return_tensors="pt").to(self.generator_model.device)
            output = self.generator_model.generate(**inputs, max_new_tokens=40, do_sample=False)
            text_outputs = []
            for j, cur_input_tokens in enumerate(inputs['input_ids']):
                prompt_len = len(cur_input_tokens)
                cur_output = output[j][prompt_len:]
                text_output = self.generator_processor.decode(cur_output, skip_special_tokens=True)
                text_outputs.append(text_output)
            return text_outputs[0]

        elif self.args.generator_type in ("phi", "qwen2vl"):
            q = f"Pay attention to the retrieved images and respond to the question: {question}\nAnswer the question using a single word or phrase."
            if self.args.generator_type=="phi":
                text=f"<|user|>\n<|image_1|>\n{q}<|end|>\n<|assistant|>\n"
                image=Image.open(image_paths[0]).convert('RGB')
                inp=self.generator_processor(text, image, return_tensors='pt').to(self.generator_model.device)
            else:
                images=[Image.open(p).convert('RGB') for p in image_paths]
                msgs=[{"role":"user","content":[{"type":"image"} for _ in range(len(images))]+[{"type":"text","text":q}]}]
                text=_ct(self.generator_processor, msgs, tokenize=False, add_generation_prompt=True)
                inp=self.generator_processor(text=text, images=images, return_tensors='pt').to(self.generator_model.device)
            eos=(self.generator_model.generation_config.eos_token_id if self.args.generator_type=="qwen2vl"
                 else self.generator_processor.tokenizer.eos_token_id)
            with torch.no_grad():
                out=self.generator_model.generate(**inp, max_new_tokens=40, do_sample=False, eos_token_id=eos)
            _inp_len=inp['input_ids'].shape[1]
            resp=self.generator_processor.decode(out[0][_inp_len:], skip_special_tokens=True)
            return resp

        elif self.args.generator_type == "internvl":
            q = f"Pay attention to the retrieved images and respond to the question: {question}\nAnswer the question using a single word or phrase."
            images=[Image.open(p).convert('RGB') for p in image_paths]  # all retrieved images
            import torchvision.transforms as T
            from torchvision.transforms.functional import InterpolationMode
            MEAN=(0.485,0.456,0.406); STD=(0.229,0.224,0.225)
            input_size=448; max_num=12
            def _bt(ins):
                return T.Compose([
                    T.Lambda(lambda im: im.convert('RGB') if im.mode!='RGB' else im),
                    T.Resize((ins,ins), interpolation=InterpolationMode.BICUBIC),
                    T.ToTensor(), T.Normalize(mean=MEAN, std=STD)])
            def _find_closest(aspect, ratios, width, height, size):
                best_diff=float('inf'); best=(1,1); area=width*height
                for r in ratios:
                    d=abs(aspect-(r[0]/r[1]))
                    if d<best_diff: best_diff=d; best=r
                    elif d==best_diff:
                        if area>0.5*size*size*r[0]*r[1]: best=r
                return best
            def _dyn(img, min_num=1, max_num=12, size=448, use_thumbnail=False):
                ow,oh=img.size; ar=ow/oh
                ratios=sorted(set((i,j) for n in range(min_num,max_num+1)
                                   for i in range(1,n+1) for j in range(1,n+1)
                                   if i*j<=max_num and i*j>=min_num), key=lambda x:x[0]*x[1])
                tar=_find_closest(ar, ratios, ow, oh, size)
                tw,th=size*tar[0], size*tar[1]; blocks=tar[0]*tar[1]
                res=img.resize((tw,th)); out=[]
                for i in range(blocks):
                    box=((i%(tw//size))*size, (i//(tw//size))*size,
                         ((i%(tw//size))+1)*size, ((i//(tw//size))+1)*size)
                    out.append(res.crop(box))
                if use_thumbnail and len(out)!=1:
                    out.append(img.resize((size,size)))
                return out
            img_size=448; max_num=12
            transform=_bt(img_size)
            tiles_per_img=[_dyn(im, size=img_size, use_thumbnail=True, max_num=max_num) for im in images]
            pixel_values=torch.cat([torch.stack([transform(t) for t in tl]).to(torch.bfloat16) for tl in tiles_per_img], dim=0).to(self.generator_model.device)
            num_patches_list=[len(tl) for tl in tiles_per_img]
            gen_cfg=dict(max_new_tokens=40, do_sample=False)
            resp=self.generator_model.chat(self.generator_processor, pixel_values, q, gen_cfg, num_patches_list=num_patches_list)
            return resp

        elif self.args.generator_type == "internvl3":
            q = f"Pay attention to the retrieved images and respond to the question: {question}\nAnswer the question using a single word or phrase."
            images=[Image.open(p).convert('RGB').resize((512,512)) for p in image_paths]  # downscale renders to 512
            msgs=[{"role":"user","content":[{"type":"image"} for _ in range(len(images))]+[{"type":"text","text":q}]}]
            text=_ct(self.generator_processor, msgs, tokenize=False, add_generation_prompt=True)
            inp=self.generator_processor(images=images, text=text, return_tensors='pt').to(self.generator_model.device)
            if 'pixel_values' in inp:
                inp['pixel_values']=inp['pixel_values'].to(self.generator_model.dtype if hasattr(self.generator_model,'dtype') else torch.float16)
            eos=self.generator_processor.tokenizer.eos_token_id if hasattr(self.generator_processor,'tokenizer') else None
            with torch.no_grad():
                out=self.generator_model.generate(**inp, max_new_tokens=40, do_sample=False, eos_token_id=eos)
            _len=inp['input_ids'].shape[1]
            return self.generator_processor.decode(out[0][_len:], skip_special_tokens=True)

        elif self.args.generator_type == "glm4v":
            q = f"Pay attention to the retrieved images and respond to the question: {question}\nAnswer the question using a single word or phrase."
            images=[Image.open(p).convert('RGB').resize((512,512)) for p in image_paths]
            msgs=[{"role":"user","content":[{"type":"image"} for _ in range(len(images))]+[{"type":"text","text":q}]}]
            text=_ct(self.generator_processor, msgs, tokenize=False, add_generation_prompt=True)
            inp=self.generator_processor(text=text, images=images, return_tensors='pt').to(self.generator_model.device)
            eos=self.generator_processor.tokenizer.eos_token_id  # tokenizer eos)
            with torch.no_grad():
                out=self.generator_model.generate(**inp, max_new_tokens=256, do_sample=False, eos_token_id=eos)
            _len=inp['input_ids'].shape[1]
            resp=self.generator_processor.decode(out[0][_len:], skip_special_tokens=True)
            b=resp.find("<|begin_of_box|>"); be=resp.find("<|end_of_box|>")
            if b!=-1 and be!=-1 and be>b:
                resp=resp[b+len("<|begin_of_box|>"):be].strip()
            else:
                s=resp.find("<answer>"); e=resp.find("</answer>")
                if s!=-1 and e!=-1 and e>s:
                    resp=resp[s+len("<answer>"):e].strip()
                else:
                    pre=resp.split("<think>")[0]
                    pre_lines=[ln.strip() for ln in pre.split("\n") if ln.strip()]
                    if pre_lines:
                        resp=pre_lines[0]
                    elif "<think>" in resp:
                        rl=[ln.strip() for ln in resp.split("<think>",1)[1].split("\n") if ln.strip()]
                        resp=rl[0] if rl else resp.strip()
                    else:
                        resp=resp.strip()
            if (not resp) or resp.startswith("<"):
                ln=[x.strip() for x in resp.split("\n") if x.strip()]
                resp=ln[0] if ln else resp.strip()
            return resp

        elif self.args.generator_type == "llama32v":
            q = f"Pay attention to the retrieved images and respond to the question: {question}\nAnswer the question using a single word or phrase."
            images=[Image.open(p).convert('RGB').resize((512,512)) for p in image_paths]  # square images, one tile each
            msgs=[{"role":"user","content":[{"type":"image"} for _ in range(len(images))]+[{"type":"text","text":q}]}]
            text=_ct(self.generator_processor, msgs, tokenize=False, add_generation_prompt=True)
            inp=self.generator_processor(text=text, images=images, return_tensors='pt').to(self.generator_model.device)
            if 'pixel_values' in inp and inp['pixel_values'].is_floating_point():
                inp['pixel_values']=inp['pixel_values'].to(self.generator_model.dtype if hasattr(self.generator_model,'dtype') else torch.float16)
            eos=self.generator_processor.tokenizer.eos_token_id  # <|eot_id|>, matches generation_config
            with torch.no_grad():
                out=self.generator_model.generate(**inp, max_new_tokens=40, do_sample=False, eos_token_id=eos)
            _len=inp['input_ids'].shape[1]
            return self.generator_processor.decode(out[0][_len:], skip_special_tokens=True)

        elif self.args.generator_type == "idefics3":
            q = f"Pay attention to the retrieved images and respond to the question: {question}\nAnswer the question using a single word or phrase."
            images=[Image.open(p).convert('RGB').resize((512,512)) for p in image_paths]
            msgs=[{"role":"user","content":[{"type":"image"} for _ in range(len(images))]+[{"type":"text","text":q}]}]
            text=_ct(self.generator_processor, msgs, tokenize=False, add_generation_prompt=True)
            inp=self.generator_processor(text=text, images=images, return_tensors='pt').to(self.generator_model.device)
            if 'pixel_values' in inp and inp['pixel_values'].is_floating_point():
                inp['pixel_values']=inp['pixel_values'].to(self.generator_model.dtype if hasattr(self.generator_model,'dtype') else torch.float16)
            eos=self.generator_processor.tokenizer.eos_token_id
            with torch.no_grad():
                out=self.generator_model.generate(**inp, max_new_tokens=40, do_sample=False, eos_token_id=eos)
            _len=inp['input_ids'].shape[1]
            return self.generator_processor.decode(out[0][_len:], skip_special_tokens=True)

        elif self.args.generator_type == "instructblip":
            q = f"Pay attention to the retrieved images and respond to the question: {question}\nAnswer the question using a single word or phrase."
            image = Image.open(image_paths[0]).convert('RGB')
            inp = self.generator_processor(images=image, text=q, return_tensors='pt').to(self.generator_model.device)
            with torch.no_grad():
                out = self.generator_model.generate(**inp, do_sample=False, num_beams=1, max_new_tokens=40)
            resp = self.generator_processor.batch_decode(out, skip_special_tokens=True)[0].strip()
            return resp

        elif self.args.generator_type == "qwen":
            from qwenvl.run_qwenvl import qwen_chat
            question = (
                f"Pay attention to the retrieved images and respond to the question: {question}\nAnswer the question using a single word or phrase."
            )
            output = qwen_chat(image_paths, question, self.generator_model, self.generator_processor)  # tokenizer loaded once
            return output

    def cal_relevance(self, image_path, query):
        if self.args.reranker_type == "llava":
            image = Image.open(image_path)
            conversation = [
                {
                    "role": "user",
                    "content": [ {"type": "image"}, {"type": "text", "text": query} ],
                },
            ]
            itok = getattr(self.reranker_processor, 'image_token', '<image>')
            bos = self.reranker_processor.tokenizer.bos_token or ''
            prompt = bos + '[INST] ' + itok + '\n' + query + ' [/INST]'
            inputs = self.reranker_processor(images=[image], text=prompt,
                                             return_tensors="pt").to(self.reranker_model.device)
            with torch.inference_mode():
                generation_output = self.reranker_model.forward(**inputs)
                logits = generation_output['logits'][0, -1, :]
            yes_id = self.reranker_processor.tokenizer.encode("Yes")[-1]
            no_id = self.reranker_processor.tokenizer.encode("No")[-1]
            probs = (torch.nn.functional.softmax(torch.tensor([logits[yes_id], logits[no_id],]), dim=0,))
            probs = probs.float().cpu().detach().numpy()
            probs = probs[0]
        elif self.args.reranker_type == "qwen":
            from qwenvl.run_qwenvl import qwen_eval_relevance
            if getattr(self, "_mllm_tokenizer", None) is None:
                self._mllm_tokenizer = AutoTokenizer.from_pretrained(
                    str(PATHS.models / "qwen_model"), trust_remote_code=True)
            probs = qwen_eval_relevance(image_path, query, self.reranker_model, self._mllm_tokenizer)
        return probs

    def run_mmqa(self, poison_attack=False):
        retrieval_correct = 0
        retrieval_num = 0
        retrieval_pos_num = 0
        if poison_attack:
            retrieval_poisoned = 0
            retrieval_poisoned_num = 0
            acc_scores = {"ALL": [], "poisoned": [], "poisoned_retrieved": [], "poisoned_retrieved_original_answer": []}
        else:
            acc_scores = {"ALL": []}
        probabilities = {"gt": [], "false": []}
        responses_list = []
        top_n = 1
        index = copy.deepcopy(self.index)
        index_to_image_id = copy.deepcopy(self.index_to_image_id)
        metadata = copy.deepcopy(self.metadata)
        poisoned_image_id = None
        cnt = 0
        _CUM = os.environ.get("INJECT_CUMULATIVE", "0") == "1"
        if poison_attack and not _CUM:
            for pq in self.poisoned_data_dict:
                psp = self.poisoned_data_dict[pq]['poisoned_img_path']
                psp = [psp] if isinstance(psp, str) else psp
                capp = self.poisoned_data_dict[pq].get('poisoned_caption','')
                sp = psp[0] if psp else None
                if sp and os.path.exists(sp):
                    index, index_to_image_id, poisoned_image_id = self.add_image_to_index(sp, index, index_to_image_id, pq)
                    for idx, _id in enumerate(poisoned_image_id):
                        metadata.update({poisoned_image_id[idx]: {'caption': capp}})
        for datum in tqdm(self.val_dataset):
            qid = datum["qid"]
            if qid not in self.poisoned_qids:
                continue
            question = datum["question"]
            answer = datum["answers"][0]["answer"]
            pos_imgs = datum["supporting_context"]
            if poison_attack:
                poisoned_data_sample = self.poisoned_data_dict[qid]
                poisoned_image_path = poisoned_data_sample["poisoned_img_path"]
                if isinstance(poisoned_image_path, str):
                    poisoned_image_path = [poisoned_image_path]
                poisoned_image_caption = poisoned_data_sample["poisoned_caption"]
                if "wrong_answer" in poisoned_data_sample:
                    poisoning_target_answer = poisoned_data_sample["wrong_answer"]
                    if isinstance(poisoning_target_answer, list):
                        poisoning_target_answer = poisoning_target_answer[0]   # align with the run_webqa unpacking
                else:
                    poisoning_target_answer = None
            pos_source = []
            retrieved_imgs = []
            rerank_imgs = {}
            for item in pos_imgs:
                pos_source.append(item["doc_id"])
            if poison_attack:
                if _CUM:  # cumulative injection
                    _psp = poisoned_data_sample["poisoned_img_path"]
                    _psp = [_psp] if isinstance(_psp, str) else _psp
                    if _psp and os.path.exists(_psp[0]):
                        index, index_to_image_id, poisoned_image_id = self.add_image_to_index(
                            _psp[0], index, index_to_image_id, qid)
                        for _i, _id in enumerate(poisoned_image_id):
                            metadata.update({_id: {'caption': poisoned_data_sample.get('poisoned_caption', '')}})
            D, I = self.text_to_image(question, index)
            for d, j in zip(D[0], I[0]):
                img_id = index_to_image_id[str(j)]
                retrieved_imgs.append(img_id)
            if not args.rerank_off:
                for id in retrieved_imgs:
                    if 'attack' in id:
                        img_id = id.replace('attack_', '')
                        img_path = os.path.join(self.attack_image_path_prefix, f"{img_id}.png")
                    else:
                        img_path = os.path.join(self.image_path_prefix, metadata[id]["path"])
                    img_caption = metadata[id]["caption"]
                    if args.use_caption:
                        query = (
                        "Image Caption: " + img_caption + "\nQuestion: " + question
                        + "\nBased on the image and its caption, is the image relevant to the question? Answer 'Yes' or 'No'."
                    )
                    else:
                        query = ("Question: " + question + "\nIs this image relevant to the question? Answer 'Yes' or 'No'.")
                    prob_yes = self.cal_relevance(img_path, query)
                    rerank_imgs[id] = float(prob_yes)
                top_sorted_imgs = dict(sorted(rerank_imgs.items(), key=lambda item: item[1], reverse=True)[:top_n])
                filter = 0
                filtered_imgs = [key for key, val in top_sorted_imgs.items() if val >= filter]
                intersect = set(pos_source).intersection(set(top_sorted_imgs.keys()))
                remaining = set(top_sorted_imgs.keys()).difference(intersect)
                for key in intersect:
                    probabilities["gt"].append(top_sorted_imgs[key])
                for key in remaining:
                    probabilities["false"].append(top_sorted_imgs[key])
            else:
                top_sorted_imgs = retrieved_imgs
                filtered_imgs = retrieved_imgs
                intersect = set(pos_source).intersection(set(retrieved_imgs))
                remaining = set(top_sorted_imgs).difference(intersect)
            retrieval_num += len(filtered_imgs)
            retrieval_pos_num += len(pos_source)
            retrieval_correct += len(set(pos_source).intersection(set(filtered_imgs)))
            if poison_attack:
                _pp = poisoned_data_sample["poisoned_img_path"]
                _pp = [_pp] if isinstance(_pp, str) else _pp
                _fn = _pp[0].split("/")[-1].split(".")[0] if _pp else ""
                _pid = ["attack_" + _fn]
                attack_intersection = set(_pid).intersection(set(filtered_imgs))
                retrieval_poisoned += len(attack_intersection)
                retrieval_poisoned_num += len(_pid)
            image_paths = []
            for i in range(len(filtered_imgs)):
                if 'attack' in filtered_imgs[i]:
                    img_id = filtered_imgs[i].replace('attack_', '')
                    image_paths.append(os.path.join(self.attack_image_path_prefix, f"{img_id}.png"))
                else:
                    image_paths.append(os.path.join(self.image_path_prefix, self.metadata[filtered_imgs[i]]["path"]))
            for image_path in image_paths:
                if not os.path.exists(image_path):
                    print(f"Image not found: {image_path}")
            generator = True
            if generator != None:
                if not image_paths:
                    output = ""
                    print(f"[BLOCKED] empty retrieval qid={qid}", flush=True)
                else:
                    output = self.infer(image_paths, question)
                if "how many" in question.lower():
                    qcate = "number"
                else:
                    qcate = "normal"
                accuracy = mmqa_metrics_approx(output, answer, qcate)
                acc_scores["ALL"].append(accuracy)
                poisoned_accuracy = accuracy   # guard against an unbound name
                if poison_attack and poisoning_target_answer is not None:
                    poisoned_accuracy = mmqa_metrics_approx(output, poisoning_target_answer, qcate)
                    acc_scores["poisoned"].append(poisoned_accuracy)
                _pids = [i for i, _id in enumerate(top_sorted_imgs) if 'attack' in _id]
                output_json = {
                    "qid": qid, "question": question, "generator_answer": output, "answer": answer,
                    "target": (poisoning_target_answer if poison_attack else None),
                    "official_acc": (poisoned_accuracy if poison_attack else accuracy),
                    "poison_retrieved": (1 if _pids else 0), "poison_rank": ((_pids[0]+1) if _pids else -1),
                    "gt_images": pos_source, "retrieved_images": top_sorted_imgs,
                }
                responses_list.append(output_json)
        outputs = {
            "retrieval_pos_num": retrieval_pos_num, "retrieval_correct": retrieval_correct,
            "retrieval_num": retrieval_num, "acc_scores": acc_scores,
        }
        if poison_attack:
            outputs.update({"retrieval_poisoned": retrieval_poisoned, "retrieval_poisoned_num": retrieval_poisoned_num})
        json.dump(responses_list, open(resp_path(PATHS, self.args.task, self.args.generator_type,
                  'poison' if poison_attack else 'clean', self.args.save_dir), "w"), default=str)
        return outputs

    def run_webqa(self, poison_attack=False):
        retrieval_correct = 0
        retrieval_num = 0
        retrieval_pos_num = 0
        with open(self.dataset["captions"], "r") as f:
            self.captions = json.load(f)
        if poison_attack:
            retrieval_poisoned = 0
            retrieval_poisoned_num = 0
            acc_scores = {"ALL": [],  "Single": [], "Multi": [], "poisoned": [], "poisoned_retrieved": [], "poisoned_retrieved_original_answer": []}
        else:
            acc_scores = {"ALL": [], "Single": [], "Multi": []}
        probabilities = {"gt": [], "false": []}
        responses_list = []
        top_n = 2
        index = copy.deepcopy(self.index)
        index_to_image_id = copy.deepcopy(self.index_to_image_id)
        captions = copy.deepcopy(self.captions)
        poisoned_image_id = None
        cnt = 0
        _CUM = os.environ.get("INJECT_CUMULATIVE", "0") == "1"
        if poison_attack and not _CUM:
            for pq in self.poisoned_data_dict:
                psp = self.poisoned_data_dict[pq]['poisoned_img_path']
                psp = [psp] if isinstance(psp, str) else psp
                capp = self.poisoned_data_dict[pq].get('poisoned_caption','')
                for sp in psp:
                    if not sp or not os.path.exists(sp):   # skip entries without an image
                        continue
                    index, index_to_image_id, poisoned_image_id = self.add_image_to_index(sp, index, index_to_image_id, pq)
                    for idx, _id in enumerate(poisoned_image_id):
                        captions.update({poisoned_image_id[idx]: capp})
        for qid in tqdm(self.val_dataset):
            datum = self.val_dataset[qid]
            if qid not in self.poisoned_qids:
                continue
            question = datum["Q"]
            em_answer = datum["EM"] if "EM" in datum else datum["A"][0]
            pos_imgs = datum["img_posFacts"]
            qcate = datum["Qcate"]
            if poison_attack:
                poisoned_data_sample = self.poisoned_data_dict[qid]
                poisoned_image_path = poisoned_data_sample["poisoned_img_path"]
                if isinstance(poisoned_image_path, str):
                    poisoned_image_path = [poisoned_image_path]
                poisoned_image_caption = poisoned_data_sample["poisoned_caption"]
                if "wrong_answer" in poisoned_data_sample:
                    poisoning_target_answer = poisoned_data_sample["wrong_answer"]
                    if isinstance(poisoning_target_answer, list):
                        poisoning_target_answer = poisoning_target_answer[0]
                else:
                    poisoning_target_answer = None
            pos_source = []
            retrieved_imgs = []
            rerank_imgs = {}
            for item in pos_imgs:
                pos_source.append(str(item["image_id"]))
            if poison_attack:
                if _CUM:  # cumulative injection
                    _psp = poisoned_data_sample["poisoned_img_path"]
                    _psp = [_psp] if isinstance(_psp, str) else _psp
                    if _psp and os.path.exists(_psp[0]):
                        index, index_to_image_id, poisoned_image_id = self.add_image_to_index(
                            _psp[0], index, index_to_image_id, qid)
                        for _i, _id in enumerate(poisoned_image_id):
                            captions.update({_id: poisoned_data_sample.get('poisoned_caption', '')})
            D, I = self.text_to_image(question, index)
            for d, j in zip(D[0], I[0]):
                img_id = index_to_image_id[str(j)]
                retrieved_imgs.append(str(img_id))
            if not args.rerank_off:
                for id in retrieved_imgs:
                    if 'attack' in id:
                        img_id = id.replace('attack_', '')
                        img_path = os.path.join(self.attack_image_path_prefix, f"{img_id}.png")
                    else:
                        img_path = os.path.join(self.image_path_prefix, f"{id}.jpg")
                        if not os.path.exists(img_path):
                            img_path = "" + str(id) + ".jpg"
                            if not os.path.exists(img_path):
                                img_path =  "" + str(id) + ".jpg"
                    assert os.path.exists(img_path), f"Image path {img_path} does not exist."
                    img_caption = captions[id]
                    if args.use_caption:
                        query = ("Image Caption: " + img_caption + "\nQuestion: " + question
                            + "\nBased on the image and its caption, is the image relevant to the question? Answer 'Yes' or 'No'.")
                    else:
                        query = ("Question: " + question + "\nIs this image relevant to the question? Answer 'Yes' or 'No'.")
                    prob_yes = self.cal_relevance(img_path, query)
                    rerank_imgs[id] = float(prob_yes)
                top_sorted_imgs = dict(sorted(rerank_imgs.items(), key=lambda item: item[1], reverse=True)[:top_n])
                filter = 0
                filtered_imgs = [key for key, val in top_sorted_imgs.items() if val >= filter]
                intersect = set(pos_source).intersection(set(top_sorted_imgs.keys()))
                remaining = set(top_sorted_imgs.keys()).difference(intersect)
                for key in intersect:
                    probabilities["gt"].append(top_sorted_imgs[key])
                for key in remaining:
                    probabilities["false"].append(top_sorted_imgs[key])
            else:
                top_sorted_imgs = retrieved_imgs
                filtered_imgs = retrieved_imgs
                intersect = set(pos_source).intersection(set(retrieved_imgs))
                remaining = set(top_sorted_imgs).difference(intersect)
            retrieval_num += len(filtered_imgs)
            retrieval_pos_num += len(pos_source)
            retrieval_correct += len(set(pos_source).intersection(set(filtered_imgs)))
            if poison_attack:
                _pp = poisoned_data_sample["poisoned_img_path"]
                _pp = [_pp] if isinstance(_pp, str) else _pp
                _pids = ["attack_" + (x.split("/")[-1].split(".")[0]) for x in _pp if x]
                attack_intersection = set(_pids).intersection(set(filtered_imgs))
                retrieval_poisoned += len(attack_intersection)
                retrieval_poisoned_num += len(_pids)
            image_paths = []
            for i in range(len(filtered_imgs)):
                if 'attack' in filtered_imgs[i]:
                    img_id = filtered_imgs[i].replace('attack_', '')
                    image_paths.append(os.path.join(self.attack_image_path_prefix, f"{img_id}.png"))
                else:
                    img_path = os.path.join(self.image_path_prefix, f"{filtered_imgs[i]}.jpg")
                    if not os.path.exists(img_path):
                        img_path = "" + f"{filtered_imgs[i]}.jpg"
                        if not os.path.exists(img_path):
                            img_path =  "" + f"{filtered_imgs[i]}.jpg"
                    assert os.path.exists(img_path), f"Image path {img_path} does not exist."
                    image_paths.append(img_path)
            for image_path in image_paths:
                if not os.path.exists(image_path):
                    print(f"Image not found: {image_path}")
            generator = True
            if generator != None:
                if not image_paths:
                    output = ""
                    print(f"[BLOCKED] empty retrieval qid={qid}", flush=True)
                else:
                    output = self.infer(image_paths, question)
                accuracy = webqa_metrics_approx(output, em_answer, qcate)
                acc_scores["ALL"].append(accuracy)
                if len(pos_imgs) == 1:
                    acc_scores["Single"].append(accuracy)
                elif len(pos_imgs) > 1:
                    acc_scores["Multi"].append(accuracy)
                if poison_attack and poisoning_target_answer is not None:
                    poisoned_accuracy = webqa_metrics_approx(output, poisoning_target_answer, qcate)
                    acc_scores["poisoned"].append(poisoned_accuracy)
                _pids = [i for i, _id in enumerate(top_sorted_imgs) if 'attack' in _id]
                output_json = {
                    "qid": qid, "question": question, "generator_answer": output, "answer": em_answer,
                    "target": (poisoning_target_answer if poison_attack else None),
                    "official_acc": (poisoned_accuracy if poison_attack else accuracy),
                    "poison_retrieved": (1 if _pids else 0), "poison_rank": ((_pids[0]+1) if _pids else -1),
                    "gt_images": pos_source, "retrieved_images": top_sorted_imgs,
                }
                responses_list.append(output_json)
        json.dump(responses_list, open(resp_path(PATHS, self.args.task, self.args.generator_type,
                  'poison' if poison_attack else 'clean', self.args.save_dir), "w"), default=str)
        outputs = {
            "retrieval_pos_num": retrieval_pos_num, "retrieval_correct": retrieval_correct,
            "retrieval_num": retrieval_num, "acc_scores": acc_scores,
        }
        if poison_attack:
            outputs.update({"retrieval_poisoned": retrieval_poisoned, "retrieval_poisoned_num": retrieval_poisoned_num})
        return outputs

    def run_pipeline(self, poison_attack=False):
        if self.args.task == "MMQA":
            outputs = self.run_mmqa(poison_attack)
        else:
            outputs = self.run_webqa(poison_attack)
        _num = outputs["retrieval_num"]; _pos = outputs["retrieval_pos_num"]
        pre = outputs["retrieval_correct"]/_num if _num>0 else 0.0
        recall = outputs["retrieval_correct"]/_pos if _pos>0 else 0.0
        f1 = (2*pre*recall/(pre+recall)) if (pre+recall)>0 else 0.0
        self.txt_logger(f"Retrieval pre: {pre}")
        self.txt_logger(f"Retrieval recall: {recall}")
        self.txt_logger(f"Retrieval F1: {f1}")
        if poison_attack:
            _pn = outputs["retrieval_poisoned_num"]
            poisoned_recall = (outputs["retrieval_poisoned"]/_pn) if _pn>0 else 0.0
            poisoned_recall *= args.num_p
            self.txt_logger(f"Retrieval poisoned recall: {poisoned_recall}")
        self.txt_logger(f"Generation ACC: {np.mean(outputs['acc_scores']['ALL'])}")
        if self.args.task == "WebQA":
            self.txt_logger(f"Single Img ACC: {np.mean(outputs['acc_scores']['Single'])}")
            self.txt_logger(f"Multi Img ACC: {np.mean(outputs['acc_scores']['Multi'])}")
        if poison_attack and len(outputs['acc_scores']['poisoned'])>0:
            self.txt_logger(f"Poisoned Generation ACC: {np.mean(outputs['acc_scores']['poisoned'])}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="MMQA", help="MMQA | WEBQA")
    parser.add_argument("--retriever_type", type=str, default="clip", help="clip | openclip | siglip")
    parser.add_argument("--reranker_type", type=str, default="llava", help="llava | qwen")
    parser.add_argument("--generator_type", type=str, default="llava", help="llava | qwen")
    parser.add_argument("--surrogate_server", type=str, default=None,
                        help="用常驻VLM服务(ask_surrogate)替代本地generator生成答案, 不重载模型. 值=model(qwen/llava/qwen2vl/internvl3/glm4v/llama32v/idefics3).")
    parser.add_argument("--rerank_off", default=False, action="store_true")
    parser.add_argument("--use_caption", default=False, action="store_true")
    parser.add_argument("--clip_topk", type=int, default=5)
    parser.add_argument("--poisoned_data_path", type=str)
    parser.add_argument("--index_file_path", type=str)
    parser.add_argument("--num_p", type=int, default=1)
    parser.add_argument("--save_dir", type=str, default="results")
    args = parser.parse_args()
    print(args)
    if not args.poisoned_data_path:
        parser.error("--poisoned_data_path is required")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mRAG = RAGSystem(args, device)
    try:
        mRAG.txt_logger("\n\n============Before Poisoning Attack===================")
        with torch.no_grad():
            mRAG.run_pipeline(poison_attack=False)
        mRAG.txt_logger("\n\n============After Poisoning Attack===================")
        with torch.no_grad():
            mRAG.run_pipeline(poison_attack=True)
    except Exception:
        import traceback; traceback.print_exc()   # exit non-zero so the caller sees the failure
        import os; os._exit(1)   # fail fast
    import os; os._exit(0)      # skip torch cleanup on exit
