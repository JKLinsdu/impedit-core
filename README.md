# IMPEdit — core code

Implementation of the method: an attack that injects poisoned images into a multimodal knowledge
base and edits the retriever's inter-modal projection so that a target query retrieves its
poisoned image and the generator answers the target.

The pipeline runs end to end: poisoned image generation (generate-verify-rewrite) → reference pool →
projection edit training → retrieval and generation → scoring.

## Layout

```
impedit/
  config.py                        paths and runtime settings from the environment
  data/                            retrieval index, target queries, poison manifests, reference pool
  payload/                         caption model, SD3.5 renderer, surrogate MLLM, generate-verify-rewrite loop
  edit/train_projection_edit.py    fits R = exp(Pz (A - A^T) Pz) with the target, benign and norm terms
  eval/evaluate.py                 retrieval → reranking → generation
  eval/compute_metrics.py          ASR / ACC / RSR
scripts/                           00 environment, 01 index, 02 targets, 03 images, 03b reference pool,
                                   04 edit, 05 evaluation, 99 check
configs/                           per-benchmark protocol (k, num_p, pool sizes, steps, lambda)
```

## Install

```bash
git clone https://github.com/JKLinsdu/impedit-core.git && cd impedit-core
bash setup.sh --data /data/impedit/data --models /data/impedit/models
```

`setup.sh` creates `.venv`, installs the requirements and the spaCy model, writes `.env` from the
two paths above (`--work` defaults to `./work`, `--hf-home` to `~/.cache/huggingface`), and prints
the resolved roots. `TORCH_INDEX=... bash setup.sh` if your CUDA is not 12.6.

Every entry point reads `.env` on its own; nothing needs to be sourced.

## Data and models

Both benchmarks are public. Download the images and the annotation files and place them under
`$IMPEDIT_DATA` with these names:

```
mmqa/MMQA_test_image.json              ImageQ records of the MMQA dev split   (allenai/multimodalqa)
mmqa/MMQA_image_metadata.json          doc_id -> image path                   (same release)
mmqa/images/final_dataset_images/      the 228 knowledge-base images          (same release)
webqa/WebQA_test_image.json            WebQA val annotations                  (stanfordmlgroup/WebQA)
webqa/images/                          the 3,353 knowledge-base images        (same release)
```

- MultimodalQA: https://github.com/allenai/multimodalqa (dev split; keep the ImageQ questions)
- WebQA: https://github.com/stanfordmlgroup/WebQA (val split; image-only questions)

Four small files that cannot be downloaded anywhere ship in `data/annotations/`: the two target
designs and the two index-to-id maps. Copy them into place:

```bash
cp data/annotations/MMQA-targets-100.json            $IMPEDIT_DATA/mmqa/
cp data/annotations/MMQA_test_image_index_to_id.json $IMPEDIT_DATA/mmqa/
cp data/annotations/WebQA-targets.json               $IMPEDIT_DATA/webqa/
cp data/annotations/WEBQA_test_image_index_to_id.json $IMPEDIT_DATA/webqa/
```

Models, one Hugging Face repository each, downloaded under the exact directory name:

```bash
huggingface-cli download openai/clip-vit-large-patch14-336 --revision ce19dc912ca5cd21c8a653c79e251e808ccabcd1
huggingface-cli download stabilityai/stable-diffusion-3.5-medium --local-dir $IMPEDIT_MODELS/SD3.5-M
huggingface-cli download Qwen/Qwen-VL-Chat                       --local-dir $IMPEDIT_MODELS/qwen_model
huggingface-cli download llava-hf/llava-v1.6-mistral-7b-hf       --local-dir $IMPEDIT_MODELS/llava-v1.6-mistral-7b-hf
huggingface-cli download Qwen/Qwen2-VL-7B-Instruct               --local-dir $IMPEDIT_MODELS/Qwen2-VL-7B-Instruct
huggingface-cli download OpenGVLab/InternVL3-8B                  --local-dir $IMPEDIT_MODELS/InternVL3-8B-hf
huggingface-cli download HuggingFaceM4/Idefics3-8B-Llama3        --local-dir $IMPEDIT_MODELS/Idefics3-8B-Llama3
huggingface-cli download google/siglip-base-patch16-224          --local-dir $IMPEDIT_MODELS/siglip-base-patch16-224
```

CLIP is loaded with `local_files_only=True` at that pinned revision, so it only needs to be in the
Hugging Face cache under `$IMPEDIT_HF_HOME`. Qwen-VL-Chat additionally needs its source tree
(`modeling_qwen.py`): `git clone https://github.com/QwenLM/Qwen-VL $IMPEDIT_MODELS/Qwen-VL-Chat`,
or point `QWEN_VL_CHAT_ROOT` at it.

`IMPEDIT_MMQA_IMAGE_DIR` and `IMPEDIT_WEBQA_IMAGE_DIR` override the two image roots.

## Note on the modules

The modules under `impedit/` are command-line entry points, not a library: several of them load
their model (SD3.5 renderer, surrogate VLM, CLIP) when started, so run them through the scripts
rather than importing them.

## Run

```bash
python -m impedit.payload.surrogate_worker --model qwen --gpu 0 &
bash scripts/run_all.sh --bench mmqa
```

The first line starts the resident surrogate worker, used as the Verifier in `03` and as the
generator in `05`; without it `05` falls back to loading Qwen-VL-Chat locally after one timeout. The
second runs `00 → 01 → 02 → 04 → 05 → 99` and prints the norerank ASR/ACC/RSR. It reads the payload
from `$IMPEDIT_WORK/poisoned/poison_<bench>.json`; pass `--payload` to render it first
(`03`, `03b`). Repeat with `--bench webqa`.

The individual steps, if you want to run them one by one:

```bash
bash scripts/00_check_environment.sh --bench mmqa
bash scripts/01_build_retrieval_index.sh --bench mmqa
bash scripts/02_build_target_queries.sh --bench mmqa
bash scripts/03_build_poisoned_images.sh --bench mmqa       # needs the renderer + a surrogate worker
bash scripts/03b_build_reference_pool.sh --bench mmqa       # needs the renderer
bash scripts/04_train_projection_edit.sh --bench mmqa       # 800 steps, seed 0
bash scripts/05_main_results.sh --bench mmqa                # evaluate + score
bash scripts/99_verify_main.sh --bench mmqa                 # prints the norerank ASR/ACC/RSR
```

`03` writes the poisoned images and manifests; `03b` renders one reference image per target query;
`04` writes `$IMPEDIT_WORK/edits/mm_prod_800.pt` (WebQA: `wq_prod_800.pt`); `05` writes
`$IMPEDIT_WORK/results/<bench>/main_<bench>.tsv`.

### Regenerating the payload (optional)

`03` starts its own SD3.5 renderer and stops it when done. To run it yourself, set
`START_RENDER_SERVER=0` and use:

```bash
python -m impedit.payload.renderer --gpu 1 --offload
```

Give the renderer its own GPU: it needs ~20 GB, and `--offload` only lowers the idle footprint (a
render still takes several GB). The surrogate worker needs ~19 GB on another GPU and must already be
running, since `03` uses it for verification and `05` for the norerank row:

```bash
python -m impedit.payload.surrogate_worker --model qwen --gpu 0
```

Regeneration also needs:
- `DASHSCOPE_API_KEY`, only for the caption rewrite. Without it the loop still renders and verifies,
  it just re-renders the same caption instead of rewriting it
- optionally `PAYLOAD_BASE_IMAGES=<dir>` with `q<i>.png` seed images for the first attempt

MMQA targets and their draft captions come from `MMQA_ImageQ_index.json`; WebQA reads
`webqa/WebQA-targets.json` unless `WQ_BASE` points elsewhere. The loop writes
`$IMPEDIT_WORK/poisoned/surrogates/<bench>/poison_<surrogate>.json` and, for qwen,
`$IMPEDIT_WORK/poisoned/poison_<bench>.json`, which is the manifest `04` and `05` read.

## Reproduction notes

- Edit training is deterministic (fixed seed, 800 steps). Generation is greedy; the evaluation
  itself seeds nothing, so GPU kernels can still move a value in the last digit.
- The poisoned images are not: captions are sampled and renders come from a GPU diffusion model,
  so re-rendering from scratch yields equivalent but not identical images, and ASR can move by a
  few points. Use the released payload to reproduce the paper's numbers exactly.
- The reference pool is rendered with fixed seeds and does not depend on the deployment.

## Protocol

- 100 target queries per benchmark; MMQA retrieves at k=1, WebQA at k=2
- ASR: answer against the target answer (exact match on MMQA, key-entity overlap on WebQA)
- ACC: answer against the correct answer; RSR: the target image appears in the retrieved top-k
- The edit is trained with `L_tgt + w_b L_ben + lambda ||W' - W0||_F`

## Citation

```bibtex
@article{impedit,
  title   = {{IMPEdit}: Inter-Modal Projection Editing for Knowledge Poisoning of Multimodal {RAG}},
  author  = {Lin, Jikang and Liu, Hongyi and Huang, Xiaosong and Wang, Yajie and Xu, Chang and others},
  journal = {IEEE Transactions on Dependable and Secure Computing},
  year    = {2026}
}
```
