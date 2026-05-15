from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from coe_lift.io import read_jsonl, write_json, write_jsonl
from coe_lift.metrics import (
    brier_score,
    cross_lingual_consistency,
    exact_grid_match,
    expected_calibration_error,
    linear_cka,
    linear_probe_accuracy,
    pairwise_group_cosine,
    split_accuracy,
)
from coe_lift.modeling import extract_coe_features, parse_layers
from coe_lift.modeling import GatedSoftLatentPrompt, OODGate


def get_hidden_size(model: torch.nn.Module) -> int:
    config = model.config
    hidden_size = getattr(config, "hidden_size", None)
    if hidden_size is None:
        hidden_size = getattr(config, "n_embd", None)
    if hidden_size is None:
        raise ValueError("Could not infer model hidden size from config.")
    return int(hidden_size)


def load_model(args: argparse.Namespace):
    quantization_config = None
    if args.load_in_4bit:
        from transformers import BitsAndBytesConfig

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    if args.adapter_dir:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter_dir)
    model.eval()
    return tokenizer, model


def load_latent_modules(
    args: argparse.Namespace,
    model: torch.nn.Module,
    device: torch.device,
) -> tuple[OODGate | None, GatedSoftLatentPrompt | None]:
    candidate_root = Path(args.adapter_dir).parent if args.adapter_dir else Path(args.model_artifact_dir)
    gate_path = Path(args.ood_gate_path) if args.ood_gate_path else candidate_root / "ood_gate.pt"
    latent_path = (
        Path(args.soft_latents_path) if args.soft_latents_path else candidate_root / "soft_latents.pt"
    )
    if not args.use_soft_latents or not gate_path.exists() or not latent_path.exists():
        return None, None

    gate = OODGate(feature_dim=3).to(device)
    gate.load_state_dict(torch.load(gate_path, map_location=device))
    gate.eval()

    state = torch.load(latent_path, map_location=device)
    num_tokens = int(state["latents"].shape[0])
    soft_latents = GatedSoftLatentPrompt(num_tokens, get_hidden_size(model)).to(device)
    soft_latents.load_state_dict(state)
    soft_latents.eval()
    return gate, soft_latents


@torch.no_grad()
def generate_one(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int,
    device: torch.device,
    layers: list[int],
    gate: OODGate | None = None,
    soft_latents: GatedSoftLatentPrompt | None = None,
) -> tuple[str, float, int, float]:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    generation_kwargs: dict[str, Any] = dict(inputs)
    prompt_len = inputs["input_ids"].shape[1]

    if gate is not None and soft_latents is not None:
        coe_out = model(**inputs, output_hidden_states=True, use_cache=False)
        coe = extract_coe_features(coe_out.hidden_states, inputs["attention_mask"], layers)
        gate_prob = torch.sigmoid(gate(coe.scalar_features))
        token_embeddings = model.get_input_embeddings()(inputs["input_ids"])
        inputs_embeds, attention_mask, _ = soft_latents(
            token_embeddings,
            inputs["attention_mask"],
            labels=None,
            gate_prob=gate_prob,
        )
        generation_kwargs = {"inputs_embeds": inputs_embeds, "attention_mask": attention_mask}
        prompt_len = 0

    start = time.perf_counter()
    out = model.generate(
        **generation_kwargs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        return_dict_in_generate=True,
        output_scores=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    elapsed = time.perf_counter() - start
    generated_ids = out.sequences[0][prompt_len:]
    prediction = tokenizer.decode(generated_ids, skip_special_tokens=True)
    if out.scores:
        chosen_probs = []
        for token_id, logits in zip(generated_ids, out.scores):
            probs = logits[0].float().softmax(dim=-1)
            chosen_probs.append(float(probs[int(token_id)].cpu()))
        confidence = float(np.mean(chosen_probs)) if chosen_probs else 0.0
    else:
        confidence = 0.0
    return prediction, confidence, int(generated_ids.numel()), elapsed


@torch.no_grad()
def extract_embeddings(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    rows: list[dict[str, Any]],
    layers: list[int],
    batch_size: int,
    max_len: int,
    device: torch.device,
) -> np.ndarray:
    embeddings = []
    for start in tqdm(range(0, len(rows), batch_size), desc="CoE diagnostics"):
        batch = rows[start : start + batch_size]
        prompts = [str(row["prompt"]) for row in batch]
        enc = tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        ).to(device)
        out = model(**enc, output_hidden_states=True, use_cache=False)
        coe = extract_coe_features(out.hidden_states, enc["attention_mask"], layers)
        embeddings.append(coe.pooled.detach().float().cpu().numpy())
    return np.concatenate(embeddings, axis=0) if embeddings else np.zeros((0, 1))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a CoE-LIFT model.")
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--adapter_dir", default="")
    parser.add_argument("--eval_jsonl", required=True)
    parser.add_argument("--output_json", default="outputs/eval_metrics.json")
    parser.add_argument("--predictions_jsonl", default="")
    parser.add_argument("--max_len", type=int, default=2048)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--diagnostics", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--diagnostic_batch_size", type=int, default=4)
    parser.add_argument("--layers", default="-18,-12,-6,-1")
    parser.add_argument("--load_in_4bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_soft_latents", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--model_artifact_dir", default="")
    parser.add_argument("--ood_gate_path", default="")
    parser.add_argument("--soft_latents_path", default="")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    rows = read_jsonl(args.eval_jsonl)
    if args.limit > 0:
        rows = rows[: args.limit]

    tokenizer, model = load_model(args)
    device = next(model.parameters()).device
    n_hidden_states = getattr(model.config, "num_hidden_layers", getattr(model.config, "n_layer", None))
    if n_hidden_states is not None:
        n_hidden_states = int(n_hidden_states) + 1
    layers = parse_layers(args.layers, n_hidden_states=n_hidden_states)
    gate, soft_latents = load_latent_modules(args, model, device)

    predictions = []
    for row in tqdm(rows, desc="Generate"):
        pred, conf, token_count, elapsed = generate_one(
            model,
            tokenizer,
            str(row["prompt"]),
            args.max_new_tokens,
            device,
            layers,
            gate=gate,
            soft_latents=soft_latents,
        )
        correct = exact_grid_match(pred, str(row["answer"]))
        predictions.append(
            {
                **row,
                "prediction": pred.strip(),
                "confidence": conf,
                "generated_tokens": token_count,
                "latency_s": elapsed,
                "correct": int(correct),
            }
        )

    correctness = [int(row["correct"]) for row in predictions]
    confidences = [float(row["confidence"]) for row in predictions]
    metrics: dict[str, Any] = {
        "overall_accuracy": float(np.mean(correctness)) if correctness else 0.0,
        "n": len(predictions),
        "split": split_accuracy(predictions),
        "cross_lingual": cross_lingual_consistency(predictions),
        "calibration": {
            "ece": expected_calibration_error(correctness, confidences),
            "brier": brier_score(correctness, confidences),
        },
        "efficiency": {
            "mean_generated_tokens": float(np.mean([row["generated_tokens"] for row in predictions]))
            if predictions
            else 0.0,
            "mean_latency_s": float(np.mean([row["latency_s"] for row in predictions])) if predictions else 0.0,
        },
    }

    if args.diagnostics and predictions:
        embeddings = extract_embeddings(
            model,
            tokenizer,
            predictions,
            layers,
            args.diagnostic_batch_size,
            args.max_len,
            device,
        )
        metrics["coe_diagnostics"] = {
            "group_cosine_alignment": pairwise_group_cosine(
                embeddings, [int(row["group_id"]) for row in predictions]
            ),
            "rule_probe_accuracy": linear_probe_accuracy(
                embeddings, [str(row["rule_family"]) for row in predictions]
            ),
            "language_probe_accuracy": linear_probe_accuracy(
                embeddings, [str(row["lang"]) for row in predictions]
            ),
        }
        id_mask = np.asarray(
            [str(row["split"]) in {"train_id", "test_id"} or str(row["split"]).endswith("_id") for row in predictions]
        )
        ood_mask = np.asarray([str(row["split"]).startswith("test_ood") for row in predictions])
        if id_mask.any() and ood_mask.any():
            n = min(int(id_mask.sum()), int(ood_mask.sum()))
            metrics["coe_diagnostics"]["id_ood_cka"] = linear_cka(
                embeddings[id_mask][:n],
                embeddings[ood_mask][:n],
            )

    write_json(args.output_json, metrics)
    if args.predictions_jsonl:
        write_jsonl(args.predictions_jsonl, predictions)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
