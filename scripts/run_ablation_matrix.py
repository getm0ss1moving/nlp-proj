from __future__ import annotations

import argparse
import json
from pathlib import Path


ABLATIONS = [
    {
        "name": "english_only_sft",
        "train_langs": "en",
        "lambda_coe": 0.0,
        "lambda_rank": 0.0,
        "lambda_gate": 0.0,
        "latent_tokens": 0,
        "layers": "-18,-12,-6,-1",
    },
    {
        "name": "multilingual_sft",
        "train_langs": "",
        "lambda_coe": 0.0,
        "lambda_rank": 0.0,
        "lambda_gate": 0.0,
        "latent_tokens": 0,
        "layers": "-18,-12,-6,-1",
    },
    {
        "name": "multilingual_coe",
        "train_langs": "",
        "lambda_coe": 0.05,
        "lambda_rank": 0.01,
        "lambda_gate": 0.0,
        "latent_tokens": 0,
        "layers": "-18,-12,-6,-1",
    },
    {
        "name": "multilingual_ood_gate",
        "train_langs": "",
        "lambda_coe": 0.0,
        "lambda_rank": 0.0,
        "lambda_gate": 0.1,
        "latent_tokens": 8,
        "layers": "-18,-12,-6,-1",
    },
    {
        "name": "full_coe_lift",
        "train_langs": "",
        "lambda_coe": 0.05,
        "lambda_rank": 0.01,
        "lambda_gate": 0.1,
        "latent_tokens": 8,
        "layers": "-18,-12,-6,-1",
    },
    {
        "name": "full_no_latents",
        "train_langs": "",
        "lambda_coe": 0.05,
        "lambda_rank": 0.01,
        "lambda_gate": 0.1,
        "latent_tokens": 0,
        "layers": "-18,-12,-6,-1",
    },
    {
        "name": "full_all_layers",
        "train_langs": "",
        "lambda_coe": 0.05,
        "lambda_rank": 0.01,
        "lambda_gate": 0.1,
        "latent_tokens": 8,
        "layers": "all",
    },
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Print CoE-LIFT ablation commands.")
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--train_jsonl", required=True)
    parser.add_argument("--eval_jsonl", required=True)
    parser.add_argument("--base_output_dir", default="outputs/ablations")
    parser.add_argument("--max_len", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--per_device_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    args = parser.parse_args()

    commands = []
    for ablation in ABLATIONS:
        output_dir = Path(args.base_output_dir) / ablation["name"]
        parts = [
            "accelerate launch --config_file configs/accelerate_8gpu.yaml scripts/train_coe_lift.py",
            f"--model {args.model}",
            f"--train_jsonl {args.train_jsonl}",
            f"--eval_jsonl {args.eval_jsonl}",
            f"--output_dir {output_dir}",
            f"--max_len {args.max_len}",
            f"--epochs {args.epochs}",
            f"--per_device_batch_size {args.per_device_batch_size}",
            f"--gradient_accumulation_steps {args.gradient_accumulation_steps}",
            f"--lambda_coe {ablation['lambda_coe']}",
            f"--lambda_rank {ablation['lambda_rank']}",
            f"--lambda_gate {ablation['lambda_gate']}",
            f"--latent_tokens {ablation['latent_tokens']}",
            f"--layers={ablation['layers']}",
        ]
        if ablation["train_langs"]:
            parts.append(f"--train_langs {ablation['train_langs']}")
        commands.append({"name": ablation["name"], "command": " ".join(parts)})

    print(json.dumps(commands, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
