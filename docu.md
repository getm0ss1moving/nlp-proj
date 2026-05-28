# Codex Q&A Log

This document records the main user questions and assistant answers from the CoE-LIFT/JYYOS working thread.

## 2026-05-14: CoE-LIFT Research Plan

**Q:** Propose an innovative NLP experiment around cross-lingual ability, fluid intelligence, Chain-of-Embedding latent reasoning, and OOD generalization under 8 x RTX 4090 constraints.

**A:** Proposed **CoE-LIFT: Chain-of-Embedding Aligned Language-Invariant Fluid Thought for Cross-Lingual OOD Reasoning**. The selected hypothesis was that cross-lingual OOD fluid-reasoning failures arise when latent CoE trajectories become language-dominated rather than rule-dominated. The plan included synthetic ARC/ConceptARC-style multilingual data, QLoRA training on Qwen-scale models, CoE contrastive alignment, OOD-gated soft latent tokens, split-level OOD evaluation, calibration, cross-lingual consistency, CKA/probe diagnostics, and ablations.

## 2026-05-14: Implement CoE-LIFT

**Q:** Implement the CoE-LIFT plan.

**A:** Created a runnable research scaffold with:

- synthetic multilingual ARC-style data generation;
- QLoRA/Accelerate training with CoE InfoNCE, rank loss, OOD gate, and soft latent tokens;
- evaluation for exact grid match, OOD gap, consistency, calibration, CoE probes, and CKA;
- ablation command generation;
- DeepSpeed/Accelerate configs;
- README and experiment protocol.

## 2026-05-14: 8-GPU Deployment

**Q:** Audit the code and create robust environment/setup/run scripts for an 8 x RTX 4090 remote server.

**A:** Added and hardened:

- `requirements.txt` with CUDA/4090-compatible dependency ranges;
- `setup_env.sh` to create `.venv`, install CUDA PyTorch and dependencies, configure Hugging Face mirror/transfer, and validate GPU count;
- `run_train.sh` to set CUDA/NCCL env vars, generate data if missing, and launch Accelerate + DeepSpeed while logging to `train.log`;
- model loading fixes for 4-bit per-rank device placement;
- grouped contrastive batching;
- cross-rank CoE gathering;
- relative CoE layer indices;
- default disabling of in-training generation evaluation to avoid distributed stalls.

## 2026-05-14: Model Download

**Q:** Is model downloading included in the setup?

**A:** The model is not pre-downloaded in `setup_env.sh`; it is downloaded automatically on first `run_train.sh` execution through `AutoTokenizer.from_pretrained()` and `AutoModelForCausalLM.from_pretrained()`. A manual pre-download option using `huggingface-cli download Qwen/Qwen3-8B --local-dir models/Qwen3-8B` was also provided.

## 2026-05-15: GitHub Repository Creation

**Q:** Create a GitHub repository and submit the first version of the code.

**A:** Created the local initial commit `5b6ab4a initial CoE-LIFT experiment scaffold` on `main`. GitHub CLI was installed, but GitHub OAuth login failed because the local machine could not complete the GitHub device-flow callback. The repository had no remote, so pushing to GitHub remained blocked pending authentication or a manually created remote URL.

## 2026-05-15: Download JYYOS Lecture Files

**Q:** Download all downloadable files from `https://jyywiki.cn/OS/demos/intro/crazy-os/` into `D:\learning\JYYOS\lecture3`.

**A:** Downloaded `Makefile`, `README.md`, `crazy-os.c`, `lib.h`, `mini-rv32ima.h`, `p1.c`, and `p2.c` into the requested directory.

## 2026-05-15: Explain `crazy-os.c` and `mini-rv32ima.h`

**Q:** Explain how `crazy-os.c` uses `mini-rv32ima.h` to implement a system.

**A:** Explained that `mini-rv32ima.h` is the software RISC-V CPU emulator, while `crazy-os.c` is the minimal OS shell. Each `Proc` contains a CPU state, memory, and output buffer. `proc_init()` loads binaries and initializes PC/SP. The scheduler repeatedly calls `rv32ima_step(&cpu, 1)` to execute one guest instruction per process. `ecall` is detected through `MCAUSE == 8`; `handle_ecall()` implements syscall 42 by writing characters to host stdout and manually restores state like `mret`.

## 2026-05-15: Why `p1` and `p2` Progress Differently

**Q:** Why does `p1.c` output progress slower than `p2.c` even though `crazy-os` uses `cur = (cur + 1) % n`?

**A:** Clarified that the scheduler is fair in guest instruction count, not in completed user-loop iterations. Each scheduler turn executes one RISC-V instruction. `p1` and `p2` receive similar instruction budgets, but `p1` completes fewer high-level loops because it prints faster-growing numbers (`10, 20, ..., 600`), causing more recursive `print_num()` work and more `ecall`s than `p2` (`1, 2, ..., 69`).

## 2026-05-15: Build and Disassemble `p1`/`p2`

**Q:** Explain how to compile and inspect `crazy-os`, and how to use `/main` in `less`.

**A:** Gave Ubuntu/WSL commands:

```bash
cd /mnt/d/learning/JYYOS/lecture3
sudo apt update
sudo apt install -y make gcc gcc-riscv64-unknown-elf binutils-riscv64-unknown-elf less
make
./crazy-os p1.bin p2.bin | less
riscv64-unknown-elf-objdump -d p1.elf > p1.disasm.txt
riscv64-unknown-elf-objdump -d p2.elf > p2.disasm.txt
```

Also explained that in `less`, typing `/main` and pressing Enter jumps to the next match; `n` continues forward, `N` searches backward, and `q` quits.

## 2026-05-15: Preserve ELF Files in Makefile

**Q:** `make` deletes `p1.elf` and `p2.elf`; how can they be preserved?

**A:** Explained that Make treats `.elf` files as intermediate files when only `.bin` targets are requested. Added:

```make
.PRECIOUS: %.elf
```

to `D:\learning\JYYOS\lecture3\Makefile`, so `p1.elf` and `p2.elf` are retained after `make`.

## 2026-05-15: Interpret `p1`/`p2` Diff

**Q:** Given a diff showing only `addi a5,a5,10` versus `addi a5,a5,1`, what does it prove?

**A:** It proves that the static `main` machine code is almost identical and the core loop update is one instruction in both programs. The performance difference is not because `x += 10` is statically much more expensive than `++x`, but because `p1` passes larger numbers into the shared `myprintf()`/`print_num()` path, causing more dynamic recursive calls, digit emissions, and syscalls.

## 2026-05-28: CoE-LIFT Experimental Results

**Q:** Given results showing `overall_accuracy=34%`, `test_id=60%`, `test_ood_lang=50%`, `test_ood_rule=8%`, `test_ood_surface=18%`, `group_cosine_alignment=0.937`, and `id_ood_cka=0.087`, what should be the next experimental update?

**A:** Recommended not blindly increasing training steps. The next phase should:

- run ablations: `sft_only`, `coe_only`, `gate_latent_only`, `full_coe_lift`, and `no_soft_latents`;
- fix EOS handling by training on `prompt + answer + eos_token`;
- modify data generation with a composition curriculum that includes some seen three-step rules while holding out other three-step combinations;
- add structured rule-trace auxiliary supervision such as `{"rules":[...],"grid":[...]}`;
- add probes for rule-set and rule-order prediction;
- interpret the key mechanism: high CoE alignment does not necessarily imply executable compositional generalization.

## 2026-05-28: Documentation and Automation Request

**Q:** Write every question and answer into GitHub by creating `docu.md`, and create a Codex automation task to compact every 3 days at 9:00.

**A:** Created this `docu.md` Q&A log locally. GitHub push still requires a configured remote and authenticated GitHub CLI/session. A recurring Codex automation is created separately to run every 3 days at 09:00 Asia/Singapore time.
