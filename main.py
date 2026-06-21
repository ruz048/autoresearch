"""
main.py — Autonomous LLM-pretraining research loop driven by the Claude Agent SDK.

Replaces the "spin up Claude Code interactively" workflow with a single Python
orchestrator. Clean separation of responsibilities:

    THE AGENT      only edits train.py (proposes one change per experiment).
    PYTHON (here)  does everything operational: git commit, run training as a
                   subprocess (no tool-timeout limit), read val_bpb, decide
                   keep/revert, and log to results.tsv.

Because Python runs the training (not a bounded agent tool call), experiments can
take arbitrarily long — e.g. with TIME_BUDGET raised to 1000s, a run is ~19 min,
which would blow past the SDK's 600s Bash-tool cap. Here that limit is irrelevant.

Usage:
    conda activate autoresearch
    python main.py                       # ~100 experiments on a fresh branch
    python main.py --experiments 20 --gpu 1
    python main.py --tag myrun --model claude-opus-4-8
    python main.py --no-setup            # skip setup, continue current branch
"""

import argparse
import asyncio
import datetime
import os
import re
import subprocess
import sys
from pathlib import Path

from claude_agent_sdk import (
    ClaudeSDKClient,
    ClaudeAgentOptions,
    AssistantMessage,
    TextBlock,
    ThinkingBlock,
    ToolUseBlock,
    ResultMessage,
    PermissionResultAllow,
    PermissionResultDeny,
)

REPO = Path(__file__).resolve().parent
PYTHON = sys.executable  # the env python running this file (used to run train.py)
RUN_LOG = REPO / "run.log"
RESULTS_TSV = REPO / "results.tsv"
EDITABLE = "train.py"  # the ONLY file the agent may touch

# ---------------------------------------------------------------------------
# Enforcement: the agent may ONLY edit train.py. It has no Bash at all, so it
# cannot run training, touch git, or reach the network. Python does all of that.
# ---------------------------------------------------------------------------

async def guard(tool_name, tool_input, context):
    if tool_name in ("Write", "Edit", "MultiEdit", "NotebookEdit"):
        fp = tool_input.get("file_path", "")
        try:
            in_repo = Path(fp).resolve().parent == REPO
        except (ValueError, RuntimeError):
            in_repo = False
        if not in_repo or Path(fp).name != EDITABLE:
            return PermissionResultDeny(
                message=f"Blocked: you may only edit {EDITABLE}. All other files are read-only.",
            )
        return PermissionResultAllow()
    # Read / Grep / Glob are fine; no other tools are granted.
    return PermissionResultAllow()


SYSTEM_APPEND = f"""
You are an autonomous ML research agent. Your ONLY job is to edit {EDITABLE} to
lower val_bpb (validation bits per byte). A separate Python harness does
everything else automatically AFTER you edit: it commits, runs the 5-minute-budget
(or longer) training, reads val_bpb, and keeps or reverts your change.

YOU DO NOT and CANNOT:
- run training, run git, run any shell command (you have no Bash tool),
- edit any file other than {EDITABLE},
- decide keep/discard — the harness compares val_bpb and decides.

EACH TURN:
1. You are told the current best val_bpb and recent experiment history.
2. Pick ONE concrete, well-motivated change to {EDITABLE} (architecture,
   optimizer, hyperparameters, batch size, schedule, etc.) you believe lowers
   val_bpb. Make focused, single-idea changes.
3. Apply it by editing {EDITABLE}. Do NOT modify prepare.py or the eval.
4. Reply with EXACTLY ONE final line describing the change, prefixed `CHANGE:`
   and containing NO commas (it goes into a tab-separated log). Example:
   CHANGE: raise MATRIX_LR 0.04->0.05 (more aggressive Muon step)

Honor a simplicity criterion: equal-or-better val_bpb with simpler code is a win;
tiny gains that add ugly complexity are not. Learn from the history — don't repeat
changes already shown to be worse, and try combining what worked.
""".strip()

PROPOSE_PROMPT = """
Current best val_bpb: {best}
Recent experiments (most recent last):
{history}

Propose and apply ONE change to train.py now. End with a single `CHANGE:` line.
""".strip()


# ---------------------------------------------------------------------------
# Logging (mirror this orchestrator's stdout/stderr to a file)
# ---------------------------------------------------------------------------

class _Tee:
    def __init__(self, stream, fh):
        self._stream, self._fh = stream, fh

    def write(self, data):
        self._stream.write(data)
        self._fh.write(data)
        self._fh.flush()
        return len(data)

    def flush(self):
        self._stream.flush()
        self._fh.flush()

    def __getattr__(self, name):
        return getattr(self._stream, name)


def setup_logging(log_path: Path):
    fh = open(log_path, "a", buffering=1, encoding="utf-8")
    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    fh.write(f"\n{'='*70}\n# main.py session started {stamp}\n{'='*70}\n")
    sys.stdout = _Tee(sys.stdout, fh)
    sys.stderr = _Tee(sys.stderr, fh)
    return fh


# ---------------------------------------------------------------------------
# Git / training / results helpers (all Python, no agent involvement)
# ---------------------------------------------------------------------------

def git(*args, check=True):
    return subprocess.run(["git", *args], cwd=REPO, check=check,
                          capture_output=True, text=True).stdout.strip()


def head_short():
    return git("rev-parse", "--short", "HEAD")


def train_py_changed():
    # exit code 1 => differences present
    return subprocess.run(["git", "diff", "--quiet", "--", EDITABLE],
                          cwd=REPO).returncode != 0


def parse_metrics(log_path: Path):
    """Return (val_bpb, peak_vram_mb) or (None, None) if the run crashed."""
    if not log_path.exists():
        return None, None
    text = log_path.read_text(errors="ignore")
    bpb = re.search(r"^val_bpb:\s+([\d.]+)", text, re.M)
    vram = re.search(r"^peak_vram_mb:\s+([\d.]+)", text, re.M)
    if not bpb:
        return None, None
    return float(bpb.group(1)), (float(vram.group(1)) if vram else 0.0)


async def run_training(gpu, run_timeout):
    """Run `python train.py` as a subprocess. Returns (val_bpb, peak_vram_mb, ok)."""
    env = os.environ.copy()
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    print(f"   ▶ training… (logging to {RUN_LOG.name}, gpu={gpu if gpu is not None else 'default'})")
    with open(RUN_LOG, "wb") as f:
        proc = await asyncio.create_subprocess_exec(
            PYTHON, "train.py", cwd=str(REPO), env=env,
            stdout=f, stderr=asyncio.subprocess.STDOUT,
        )
        try:
            await asyncio.wait_for(proc.wait(), timeout=run_timeout)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            print(f"   ✗ run exceeded {run_timeout}s — killed, treating as crash")
            return None, None, False
    bpb, vram = parse_metrics(RUN_LOG)
    if bpb is None:
        tail = "\n".join(RUN_LOG.read_text(errors="ignore").splitlines()[-15:])
        print(f"   ✗ crash (no val_bpb). Last lines:\n{tail}")
        return None, None, False
    return bpb, vram, True


def append_result(commit, bpb, vram, status, desc):
    if not RESULTS_TSV.exists():
        RESULTS_TSV.write_text("commit\tval_bpb\tmemory_gb\tstatus\tdescription\n")
    bpb_s = f"{bpb:.6f}" if bpb is not None else "0.000000"
    gb_s = f"{vram/1024:.1f}" if vram else "0.0"
    with RESULTS_TSV.open("a") as f:
        f.write(f"{commit}\t{bpb_s}\t{gb_s}\t{status}\t{desc}\n")


def results_history(n=12):
    if not RESULTS_TSV.exists():
        return "(none yet)"
    rows = RESULTS_TSV.read_text().splitlines()[1:]  # drop header
    return "\n".join(rows[-n:]) if rows else "(none yet)"


def setup_run(tag):
    """Pure-Python setup: verify cache, create the branch, init results.tsv."""
    cache = Path.home() / ".cache" / "autoresearch"
    if not (cache / "tokenizer" / "tokenizer.pkl").exists():
        print(f"!! data/tokenizer cache missing at {cache} — run `python prepare.py` first.")
        sys.exit(1)

    branch = f"autoresearch/{tag}"
    existing = git("branch", "--list", branch)
    if existing:
        print(f"!! branch {branch} already exists — pick a different --tag.")
        sys.exit(1)
    git("checkout", "-b", branch)
    # Anchor the exact working state (incl. any prepare.py budget change) as the
    # baseline commit so `git reset --hard` reverts are always safe.
    git("add", "-A")
    if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=REPO).returncode != 0:
        git("commit", "-m", f"baseline: run config for {branch}")
    print(f"   branch {branch} @ {head_short()} ready")
    # results.tsv starts fresh (untracked / gitignored)
    RESULTS_TSV.write_text("commit\tval_bpb\tmemory_gb\tstatus\tdescription\n")


# ---------------------------------------------------------------------------
# Agent interaction: one turn that edits train.py and returns a description
# ---------------------------------------------------------------------------

async def propose_edit(client, best, history):
    prompt = PROPOSE_PROMPT.format(best=(f"{best:.6f}" if best else "n/a"), history=history)
    await client.query(prompt)
    desc = None
    async for message in client.receive_response():
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock) and block.text.strip():
                    print(f"   🤖 {block.text.strip()[:600]}")
                    m = re.search(r"CHANGE:\s*(.+)", block.text)
                    if m:
                        desc = m.group(1).strip().replace("\t", " ").replace(",", ";")
                elif isinstance(block, ThinkingBlock):
                    print("      …thinking…")
                elif isinstance(block, ToolUseBlock) and block.name in ("Edit", "Write"):
                    print(f"      ✏️  edit {Path(block.input.get('file_path','')).name}")
        elif isinstance(message, ResultMessage):
            break
    return desc or "edit train.py (no description)"


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

async def run(args):
    options = ClaudeAgentOptions(
        cwd=str(REPO),
        model=args.model,
        system_prompt={"type": "preset", "preset": "claude_code", "append": SYSTEM_APPEND},
        allowed_tools=["Read", "Edit", "Grep", "Glob"],  # NO Bash, NO Write-elsewhere
        can_use_tool=guard,
        permission_mode="default",
        max_turns=args.max_turns,
        setting_sources=["project"],
    )

    print(f"Repo:        {REPO}")
    print(f"Model:       {args.model}")
    print(f"Branch tag:  autoresearch/{args.tag}")
    print(f"Experiments: {args.experiments}   GPU: {args.gpu}   run-timeout: {args.run_timeout}s")
    print("=" * 70)

    if not args.no_setup:
        print("\n### SETUP ###")
        setup_run(args.tag)

    best_bpb = None
    keep_commit = head_short()

    async with ClaudeSDKClient(options=options) as client:
        for n in range(1, args.experiments + 1):
            print(f"\n### EXPERIMENT {n}/{args.experiments} ###")

            if n == 1 and not args.no_setup:
                # Baseline: run train.py unchanged at the branch baseline commit.
                desc, commit = "baseline", keep_commit
            else:
                desc = await propose_edit(client, best_bpb, results_history())
                if not train_py_changed():
                    print("   ⚠️  agent did not modify train.py — skipping this experiment.")
                    continue
                git("add", EDITABLE)
                git("commit", "-m", f"exp {n}: {desc}")
                commit = head_short()
                print(f"   committed {commit}: {desc}")

            bpb, vram, ok = await run_training(args.gpu, args.run_timeout)

            if not ok:
                status = "crash"
                if commit != keep_commit:
                    git("reset", "--hard", keep_commit)
            elif best_bpb is None or bpb < best_bpb:
                status, best_bpb, keep_commit = "keep", bpb, commit
            else:
                status = "discard"
                if commit != keep_commit:
                    git("reset", "--hard", keep_commit)

            append_result(commit, bpb, vram, status, desc)
            best_s = f"{best_bpb:.6f}" if best_bpb else "n/a"
            bpb_s = f"{bpb:.6f}" if bpb else "CRASH"
            print(f"   → {status.upper()}: val_bpb={bpb_s} (best={best_s})")

    print("\n" + "=" * 70)
    print(f"Done. Best val_bpb: {best_bpb if best_bpb else 'n/a'}. See results.tsv and `git log`.")


def main():
    default_tag = datetime.datetime.now().strftime("%b%d-%H%M%S").lower()
    p = argparse.ArgumentParser(description="Autonomous autoresearch loop via Claude Agent SDK")
    p.add_argument("--experiments", type=int, default=100, help="number of experiments to run")
    p.add_argument("--tag", default=default_tag,
                   help="run tag -> branch autoresearch/<tag> (default: timestamp, e.g. jun21-143022)")
    p.add_argument("--model", default="claude-opus-4-8", help="model id for the agent")
    p.add_argument("--gpu", default=None, help="CUDA_VISIBLE_DEVICES for training (e.g. 1)")
    p.add_argument("--run-timeout", type=int, default=2400,
                   help="kill a single training run after this many seconds (hang guard)")
    p.add_argument("--max-turns", type=int, default=12, help="max agent turns per edit proposal")
    p.add_argument("--no-setup", action="store_true", help="skip setup; continue current branch")
    p.add_argument("--log-file", default=None,
                   help="orchestrator log path (default: main_<tag>.log; '-' to disable)")
    args = p.parse_args()

    if args.log_file != "-":
        log_path = Path(args.log_file) if args.log_file else REPO / f"main_{args.tag}.log"
        setup_logging(log_path)
        print(f"Logging this session to {log_path}")

    asyncio.run(run(args))


if __name__ == "__main__":
    main()
