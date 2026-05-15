# CoE-LIFT

CoE-LIFT implements the proposed experiment:

**Chain-of-Embedding Aligned Language-Invariant Fluid Thought for Cross-Lingual OOD Reasoning**.

The repository contains a reproducible scaffold for:

- generating ARC/ConceptARC-style multilingual JSONL data;
- QLoRA SFT with CoE supervised contrastive alignment;
- an OOD gate based on CoE sparsity/curvature/drift features;
- gated soft latent tokens for lightweight latent expansion;
- evaluation for exact match, OOD gap, consistency, calibration, and CoE diagnostics.

## Environment

Recommended paper environment:

- Ubuntu 22.04 or WSL2 Ubuntu
- Python 3.10 or 3.11
- CUDA 12.1 or 12.4
- 8 x NVIDIA RTX 4090 24GB

Install:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

On native Windows, install `requirements.txt` for code/data smoke tests. CUDA QLoRA with
`bitsandbytes` and `deepspeed` is expected to be run from Linux/WSL.

## Generate Data

```bash
python scripts/generate_synthetic_data.py \
  --output_dir data/synthetic \
  --train_groups 1200 \
  --eval_groups 300 \
  --seed 13
```

This creates:

- `data/synthetic/coe_lift_train.jsonl`
- `data/synthetic/coe_lift_eval.jsonl`
- `data/synthetic/metadata.json`

The schema follows the plan:

```json
{
  "id": "task_0001_zh_perm3",
  "group_id": 1,
  "lang": "zh",
  "split": "train_id",
  "prompt": "...",
  "answer": "{\"grid\":[[1,2],[2,1]]}",
  "rule_family": "rotate_cw+color_map",
  "surface_seed": 3,
  "ood_label": 0
}
```

## Train

Single-node 8 GPU run:

```bash
accelerate launch --config_file configs/accelerate_8gpu.yaml scripts/train_coe_lift.py \
  --model Qwen/Qwen3-8B \
  --train_jsonl data/synthetic/coe_lift_train.jsonl \
  --eval_jsonl data/synthetic/coe_lift_eval.jsonl \
  --output_dir outputs/coe_lift_qwen3_8b \
  --max_len 2048 \
  --epochs 2 \
  --per_device_batch_size 2 \
  --gradient_accumulation_steps 8 \
  --layers=-18,-12,-6,-1 \
  --eval_samples 0 \
  --load_in_4bit
```

For a local CPU smoke test, use a tiny model and disable QLoRA:

```bash
python scripts/generate_synthetic_data.py --output_dir data/smoke --train_groups 4 --eval_groups 2
python scripts/train_coe_lift.py \
  --model hf-internal-testing/tiny-random-GPT2LMHeadModel \
  --train_jsonl data/smoke/coe_lift_train.jsonl \
  --eval_jsonl data/smoke/coe_lift_eval.jsonl \
  --output_dir outputs/smoke \
  --epochs 1 \
  --max_steps 2 \
  --per_device_batch_size 2 \
  --gradient_accumulation_steps 1 \
  --no_load_in_4bit \
  --no_use_lora \
  --layers=-2,-1
```

## Evaluate

```bash
python scripts/evaluate_coe_lift.py \
  --model Qwen/Qwen3-8B \
  --adapter_dir outputs/coe_lift_qwen3_8b/adapter \
  --eval_jsonl data/synthetic/coe_lift_eval.jsonl \
  --output_json outputs/coe_lift_qwen3_8b/eval_metrics.json \
  --max_new_tokens 256 \
  --load_in_4bit
```

## Ablations

`scripts/run_ablation_matrix.py` prints the planned command matrix for:

- English-only SFT;
- multilingual SFT;
- multilingual SFT + CoE contrastive;
- multilingual SFT + OOD gate;
- full CoE-LIFT;
- no soft latent tokens;
- all-layer vs mid/upper-layer alignment.

```bash
python scripts/run_ablation_matrix.py \
  --train_jsonl data/synthetic/coe_lift_train.jsonl \
  --eval_jsonl data/synthetic/coe_lift_eval.jsonl
```

## Notes

The implementation intentionally keeps generation targets as canonical JSON and does not train
long natural-language CoT. This makes the latent/CoE intervention easier to isolate from ordinary
instruction tuning or explanation-style SFT.
