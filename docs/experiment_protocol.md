# CoE-LIFT Experiment Protocol

## Research Claim

CoE-LIFT tests whether cross-lingual OOD fluid-reasoning failures are visible as
language-dominated Chain-of-Embedding trajectories, and whether aligning abstractly
equivalent trajectories improves generalization under language, rule, and surface shifts.

## Data Splits

- `train_id`: English, Chinese, Spanish, and French variants of synthetic ARC-style rule tasks.
- `train_probe_surface`: shifted surface variants used only to supervise the OOD gate.
- `test_id`: held-out groups with seen languages, rules, and grid sizes.
- `test_ood_lang`: held-out languages with seen rules and grid sizes.
- `test_ood_rule`: unseen three-rule compositions with seen languages.
- `test_ood_surface`: larger grids and color/symbol remappings.

All answers are canonical JSON: `{"grid": ...}`. Prompts may be multilingual, but targets do
not contain natural-language chain-of-thought.

## Training Conditions

Primary run:

- Model: `Qwen/Qwen3-8B`
- Method: QLoRA + CoE supervised contrastive loss + OOD gate + 8 soft latent tokens
- CoE layers: `18,24,30,35`
- Loss: `L_answer + 0.05 L_coe + 0.01 L_rank + 0.1 L_gate`

Ablations:

- English-only SFT
- multilingual SFT
- multilingual SFT + CoE contrastive
- multilingual SFT + OOD gate
- full CoE-LIFT
- full CoE-LIFT without soft latent tokens
- full CoE-LIFT with all-layer alignment

## Metrics

Report the following for every run:

- exact grid match by split;
- OOD gap: ID accuracy minus OOD accuracy;
- cross-lingual group consistency;
- correctness variance within multilingual groups;
- ECE and Brier score from generation-token confidence;
- mean generated tokens and latency;
- CoE group cosine alignment;
- CoE rule probe accuracy;
- CoE language probe accuracy;
- optional ID/OOD CKA.

## Artifacts

Each training output directory should contain:

- `adapter/`: PEFT adapter or full model checkpoint;
- `tokenizer/`: tokenizer snapshot;
- `coe_projector.pt`: CoE projection head;
- `ood_gate.pt`: OOD gate;
- `soft_latents.pt`: learned latent-expansion tokens;
- `train_args.json`: reproducibility arguments;
- `training_summary.json`: final step count.

Each evaluation should write:

- `eval_metrics.json`
- optional predictions JSONL with raw generations and per-example metadata.
