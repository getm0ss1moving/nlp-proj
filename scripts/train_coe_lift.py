from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from torch.utils.data import BatchSampler, DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from coe_lift.io import read_jsonl, write_json
from coe_lift.modeling import (
    CoEProjector,
    GatedSoftLatentPrompt,
    OODGate,
    effective_rank_loss,
    extract_coe_features,
    gather_no_grad,
    gather_with_grad,
    group_info_nce,
    parse_layers,
)


class JsonlDataset(Dataset):
    def __init__(self, path: str | Path, langs: set[str] | None = None):
        rows = read_jsonl(path)
        if langs is not None:
            rows = [row for row in rows if row.get("lang") in langs]
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[index]


class GroupedContrastiveBatchSampler(BatchSampler):
    """Build batches with same-group positives and cross-group negatives."""

    def __init__(
        self,
        rows: list[dict[str, Any]],
        batch_size: int,
        seed: int,
        drop_last: bool = False,
    ) -> None:
        if batch_size < 2:
            raise ValueError("CoE grouped batches require batch_size >= 2.")
        if batch_size % 2 != 0:
            raise ValueError("CoE grouped batches require an even batch_size.")
        self.rows = rows
        self.batch_size = batch_size
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0
        self.groups: dict[int, list[int]] = {}
        for idx, row in enumerate(rows):
            self.groups.setdefault(int(row["group_id"]), []).append(idx)
        self.group_ids = [gid for gid, indices in self.groups.items() if len(indices) >= 2]
        if not self.group_ids:
            raise ValueError("Grouped batching needs at least one group_id with two records.")

    def __len__(self) -> int:
        n = sum(((len(self.groups[gid]) + 1) // 2) * 2 for gid in self.group_ids)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        pairs: list[list[int]] = []
        for gid in self.group_ids:
            candidates = self.groups[gid]
            perm = torch.randperm(len(candidates), generator=generator).tolist()
            shuffled = [candidates[idx] for idx in perm]
            if len(shuffled) % 2 == 1:
                shuffled.append(shuffled[0])
            for start in range(0, len(shuffled), 2):
                pairs.append(shuffled[start : start + 2])
        pair_order = torch.randperm(len(pairs), generator=generator).tolist()
        cursor = 0
        while cursor < len(pair_order):
            batch: list[int] = []
            while len(batch) < self.batch_size and cursor < len(pair_order):
                pair = pairs[pair_order[cursor]]
                cursor += 1
                batch.extend(pair[: max(0, self.batch_size - len(batch))])
            if len(batch) == self.batch_size or (batch and not self.drop_last):
                yield batch


def init_dist() -> None:
    if "LOCAL_RANK" in os.environ and torch.cuda.is_available():
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def str_to_langs(value: str) -> set[str] | None:
    if not value:
        return None
    return {part.strip() for part in value.split(",") if part.strip()}


def collate(batch: list[dict[str, Any]], tokenizer: AutoTokenizer, max_len: int) -> dict[str, Any]:
    prompts = [str(item["prompt"]) for item in batch]
    answers = [str(item["answer"]) for item in batch]
    texts = [prompt + answer for prompt, answer in zip(prompts, answers)]
    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
    )
    labels = enc["input_ids"].clone()
    for i, prompt in enumerate(prompts):
        prompt_ids = tokenizer(
            prompt,
            truncation=True,
            max_length=max_len,
            add_special_tokens=True,
        ).input_ids
        labels[i, : min(len(prompt_ids), labels.size(1))] = -100
    enc["labels"] = labels
    enc["group_ids"] = torch.tensor([int(item["group_id"]) for item in batch], dtype=torch.long)
    enc["ood_labels"] = torch.tensor([int(item.get("ood_label", 0)) for item in batch], dtype=torch.float)
    enc["ids"] = [str(item["id"]) for item in batch]
    return enc


@torch.no_grad()
def exact_eval(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    rows: list[dict[str, Any]],
    device: torch.device,
    max_new_tokens: int,
) -> float:
    model.eval()
    ok = 0
    for row in rows:
        inputs = tokenizer(str(row["prompt"]), return_tensors="pt").to(device)
        generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        pred = tokenizer.decode(generated[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        ok += int(pred.strip() == str(row["answer"]).strip())
    model.train()
    return ok / max(1, len(rows))


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

    device_map = None
    if args.load_in_4bit and torch.cuda.is_available():
        device_map = {"": int(os.environ.get("LOCAL_RANK", "0"))}

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map=device_map,
        trust_remote_code=True,
    )
    model.config.use_cache = False
    if args.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    if args.use_lora:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

        if args.load_in_4bit:
            model = prepare_model_for_kbit_training(model)
        target_modules = [part.strip() for part in args.target_modules.split(",") if part.strip()]
        model = get_peft_model(
            model,
            LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=target_modules,
            ),
        )
    return tokenizer, model


def get_hidden_size(model: torch.nn.Module) -> int:
    config = model.config
    hidden_size = getattr(config, "hidden_size", None)
    if hidden_size is None:
        hidden_size = getattr(config, "n_embd", None)
    if hidden_size is None:
        raise ValueError("Could not infer model hidden size from config.")
    return int(hidden_size)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train CoE-LIFT.")
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--train_jsonl", required=True)
    parser.add_argument("--eval_jsonl", required=True)
    parser.add_argument("--output_dir", default="outputs/coe_lift")
    parser.add_argument("--train_langs", default="", help="Optional comma-separated language filter.")
    parser.add_argument("--max_len", type=int, default=2048)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--per_device_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--mixed_precision", default="bf16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--layers", default="-18,-12,-6,-1")
    parser.add_argument("--coe_dim", type=int, default=256)
    parser.add_argument("--lambda_coe", type=float, default=0.05)
    parser.add_argument("--lambda_rank", type=float, default=0.01)
    parser.add_argument("--lambda_gate", type=float, default=0.1)
    parser.add_argument("--info_nce_temperature", type=float, default=0.07)
    parser.add_argument("--latent_tokens", type=int, default=8)
    parser.add_argument("--gate_threshold", type=float, default=0.5)
    parser.add_argument("--eval_samples", type=int, default=0)
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--grouped_batches", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gradient_checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_lora", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--load_in_4bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--target_modules",
        default="q_proj,k_proj,v_proj,o_proj,up_proj,down_proj,gate_proj",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    init_dist()
    torch.manual_seed(args.seed)

    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )

    output_dir = Path(args.output_dir)
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        write_json(output_dir / "train_args.json", vars(args))

    tokenizer, model = load_model(args)
    train_data = JsonlDataset(args.train_jsonl, langs=str_to_langs(args.train_langs))
    eval_rows = read_jsonl(args.eval_jsonl)
    if len(train_data) == 0:
        raise ValueError("No training rows after applying filters.")

    if args.grouped_batches:
        batch_sampler = GroupedContrastiveBatchSampler(
            train_data.rows,
            batch_size=args.per_device_batch_size,
            seed=args.seed,
        )
        train_loader = DataLoader(
            train_data,
            batch_sampler=batch_sampler,
            collate_fn=lambda batch: collate(batch, tokenizer, args.max_len),
        )
    else:
        batch_sampler = None
        train_loader = DataLoader(
            train_data,
            batch_size=args.per_device_batch_size,
            shuffle=True,
            collate_fn=lambda batch: collate(batch, tokenizer, args.max_len),
        )

    n_hidden_states = getattr(model.config, "num_hidden_layers", getattr(model.config, "n_layer", None))
    if n_hidden_states is not None:
        n_hidden_states = int(n_hidden_states) + 1
    layers = parse_layers(args.layers, n_hidden_states=n_hidden_states)
    hidden_size = get_hidden_size(model)
    projector = CoEProjector(hidden_size, len(layers), args.coe_dim)
    gate = OODGate(feature_dim=3)
    soft_latents = GatedSoftLatentPrompt(args.latent_tokens, hidden_size)

    parameters = list(model.parameters()) + list(projector.parameters()) + list(gate.parameters())
    if args.latent_tokens > 0:
        parameters += list(soft_latents.parameters())
    parameters = [parameter for parameter in parameters if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)

    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * args.epochs
    if args.max_steps > 0:
        total_steps = min(total_steps, args.max_steps)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=min(args.warmup_steps, max(1, total_steps // 10)),
        num_training_steps=max(1, total_steps),
    )

    model, projector, gate, soft_latents, optimizer, train_loader, scheduler = accelerator.prepare(
        model, projector, gate, soft_latents, optimizer, train_loader, scheduler
    )
    embed = accelerator.unwrap_model(model).get_input_embeddings()

    global_step = 0
    model.train()
    progress = tqdm(total=total_steps, disable=not accelerator.is_main_process)
    for epoch in range(args.epochs):
        if batch_sampler is not None:
            batch_sampler.set_epoch(epoch)
        for batch in train_loader:
            group_ids = batch.pop("group_ids").to(accelerator.device)
            ood_labels = batch.pop("ood_labels").to(accelerator.device)
            batch.pop("ids", None)
            labels = batch.pop("labels").to(accelerator.device)
            input_ids = batch.pop("input_ids").to(accelerator.device)
            attention_mask = batch.pop("attention_mask").to(accelerator.device)

            with accelerator.accumulate(model):
                first = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    use_cache=False,
                )
                coe = extract_coe_features(first.hidden_states, attention_mask, layers)
                z = projector(coe.pooled)
                gate_logits = gate(coe.scalar_features.detach())
                gate_prob = torch.sigmoid(gate_logits)

                token_embeddings = embed(input_ids)
                inputs_embeds, latent_attention_mask, latent_labels = soft_latents(
                    token_embeddings,
                    attention_mask,
                    labels,
                    gate_prob=gate_prob,
                )
                second = model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=latent_attention_mask,
                    labels=latent_labels,
                    output_hidden_states=False,
                    use_cache=False,
                )

                loss_answer = second.loss
                global_z = gather_with_grad(z)
                global_group_ids = gather_no_grad(group_ids)
                loss_coe = group_info_nce(
                    global_z,
                    global_group_ids,
                    temperature=args.info_nce_temperature,
                )
                loss_rank = effective_rank_loss(z)
                loss_gate = F.binary_cross_entropy_with_logits(gate_logits, ood_labels)

                loss = (
                    loss_answer
                    + args.lambda_coe * loss_coe
                    + args.lambda_rank * loss_rank
                    + args.lambda_gate * loss_gate
                )
                accelerator.backward(loss)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if accelerator.is_main_process and global_step % args.log_every == 0:
                print(
                    json.dumps(
                        {
                            "epoch": epoch,
                            "step": global_step,
                            "loss": float(loss.detach().cpu()),
                            "loss_answer": float(loss_answer.detach().cpu()),
                            "loss_coe": float(loss_coe.detach().cpu()),
                            "loss_rank": float(loss_rank.detach().cpu()),
                            "loss_gate": float(loss_gate.detach().cpu()),
                            "gate_prob_mean": float(gate_prob.detach().mean().cpu()),
                        }
                    )
                )

            global_step += 1
            progress.update(1)
            if args.max_steps > 0 and global_step >= args.max_steps:
                break
        if args.max_steps > 0 and global_step >= args.max_steps:
            break

        if args.eval_samples > 0:
            accelerator.wait_for_everyone()
        if accelerator.is_main_process and args.eval_samples > 0:
            eval_subset = eval_rows[: min(args.eval_samples, len(eval_rows))]
            acc = exact_eval(
                accelerator.unwrap_model(model),
                tokenizer,
                eval_subset,
                accelerator.device,
                args.max_new_tokens,
            )
            print(json.dumps({"epoch": epoch, "eval_exact": acc}))
        if args.eval_samples > 0:
            accelerator.wait_for_everyone()

    progress.close()
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        adapter_dir = output_dir / "adapter"
        accelerator.unwrap_model(model).save_pretrained(adapter_dir)
        tokenizer.save_pretrained(output_dir / "tokenizer")
        torch.save(accelerator.unwrap_model(projector).cpu().state_dict(), output_dir / "coe_projector.pt")
        torch.save(accelerator.unwrap_model(gate).cpu().state_dict(), output_dir / "ood_gate.pt")
        torch.save(accelerator.unwrap_model(soft_latents).cpu().state_dict(), output_dir / "soft_latents.pt")
        write_json(output_dir / "training_summary.json", {"global_step": global_step})


if __name__ == "__main__":
    main()
