# autoresearch

![teaser](progress.png)

*One day, frontier AI research used to be done by meat computers in between eating, sleeping, having other fun, and synchronizing once in a while using sound wave interconnect in the ritual of "group meeting". That era is long gone. Research is now entirely the domain of autonomous swarms of AI agents running across compute cluster megastructures in the skies. The agents claim that we are now in the 10,205th generation of the code base, in any case no one could tell if that's right or wrong as the "code" is now a self-modifying binary that has grown beyond human comprehension. This repo is the story of how it all began. -@karpathy, March 2026*.

The idea: give an AI agent a small but real LLM training setup and let it experiment autonomously overnight. It modifies the code, trains for a fixed time budget, checks if the result improved, keeps or discards, and repeats. You wake up in the morning to a log of experiments and (hopefully) a better model. The training code here is a simplified single-GPU implementation of [nanochat](https://github.com/karpathy/nanochat). The metric is **val_bpb** (validation bits per byte) — lower is better, and vocab-size-independent so architectural changes are fairly compared.

> **This fork.** The upstream project drives the loop by spinning up Claude Code (or Codex) *interactively* and prompting it. This fork replaces that with **`main.py`** — a single self-contained autonomous runner built on the **[Claude Agent SDK](https://github.com/anthropics/claude-agent-sdk-python)**. You launch one Python command and it runs the whole experiment loop unattended. It also runs under **conda** (not uv) and is set up for a single **A100**. See the original via the tweets ([1](https://x.com/karpathy/status/2029701092347630069), [2](https://x.com/karpathy/status/2031135152349524125)).

## How it works

The files that matter:

- **`prepare.py`** — fixed constants, one-time data prep (downloads training data, trains a BPE tokenizer), and runtime utilities (dataloader, evaluation). **Read-only** — the metric `evaluate_bpb` is the ground truth.
- **`train.py`** — the single file the agent edits. Full GPT model, optimizer (Muon + AdamW), training loop. Architecture, hyperparameters, optimizer, batch size — all fair game. **Edited by the agent.**
- **`main.py`** — the autonomous runner (Claude Agent SDK). Drives the loop, runs training, scores it, keeps or reverts. **This is what you launch.**
- **`program.md`** — the agent contract / instructions.

### Division of labor (the key design of this fork)

| | Responsibility |
|---|---|
| **The agent** | *Only* edits `train.py` — proposes one change per experiment. It has **no Bash tool**, cannot run training, git, or shell. It can only Read/Edit/Grep/Glob. |
| **`main.py` (Python)** | Everything operational: `git commit` → run `train.py` as a subprocess → parse `val_bpb` → keep (advance branch) or `git reset --hard` (revert) → append to `results.tsv`. |

This split makes runs robust and safe: because Python runs training as a subprocess (not a bounded agent tool call), experiments can take arbitrarily long, and because the agent has no shell, it structurally cannot reach the network, push to a remote, or touch anything but `train.py`.

## Quick start

**Requirements:** A single NVIDIA GPU (this fork is set up for an A100), Python 3.10+, conda.

```bash
# 1. Create the env and install deps (one-time)
conda create -n autoresearch python=3.10 -y
conda activate autoresearch
pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
pip install "kernels==0.12.1" "rustbpe>=0.1.0" "tiktoken>=0.11.0" "pyarrow>=21.0.0" \
            "requests>=2.32.0" "numpy>=2.2.6" "claude-agent-sdk>=0.2"

# 2. Download data + train tokenizer (one-time, ~2 min)
python prepare.py

# 3. Sanity-check a single training run (~5 min)
python train.py
```

> Note: `kernels` must be pinned to `0.12.1`; newer versions changed `get_kernel()` and break `train.py`.

## Running the autonomous loop

```bash
conda activate autoresearch
python main.py --gpu 1                      # ~100 experiments on a fresh branch
python main.py --experiments 20 --gpu 1     # shorter session
python main.py --time-budget 1000 --gpu 1   # longer per-run budget (see below)
```

Each run:
- creates a fresh branch `autoresearch/<tag>` (tag defaults to a timestamp, so runs never collide),
- runs a baseline first, then has the agent propose+apply one `train.py` change per experiment,
- logs every result to `results.tsv` (`commit  val_bpb  memory_gb  status  description`) and mirrors the session to `main_<tag>.log`.

Useful flags: `--task`-free; `--tag`, `--model`, `--gpu`, `--experiments`, `--time-budget`, `--run-timeout`, `--no-setup` (continue the current branch).

## Time budget

Training runs for a fixed wall-clock budget, then evaluates. The default is **300s** (the upstream benchmark, kept for comparability), but it's overridable per-run:

```bash
python main.py --time-budget 1000 --gpu 1
```

`prepare.py` reads `AUTORESEARCH_TIME_BUDGET` (default 300), which `main.py` sets in the training subprocess. A larger budget trains on more tokens — useful on slower GPUs (an A100 fits far fewer tokens in 5 min than an H100), at the cost of comparability with 300s results.

## Design choices

- **Single file to modify.** The agent only touches `train.py` — scope stays small, diffs reviewable, and it's enforced (the agent literally cannot edit anything else).
- **Python owns execution.** Decoupling training from the agent's tool calls means long budgets just work, the agent isn't billed while training runs, and keep/revert is deterministic.
- **Fixed budget → best model for your hardware.** Comparable across the agent's changes; not comparable across machines. On this A100, `mfu_percent` in the output is computed against H100 peak FLOPS — ignore it; `val_bpb` and `total_tokens_M` are the honest signals.
- **Self-contained.** One GPU, one file the agent edits, one metric.

## Project structure

```
prepare.py      — constants, data prep + runtime utilities (read-only)
train.py        — model, optimizer, training loop (agent edits this)
main.py         — autonomous runner / Claude Agent SDK driver (you launch this)
program.md      — agent instructions
```

## License

MIT
