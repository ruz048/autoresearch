"""
main.py — Autonomous LLM-pretraining research loop driven by the Claude Agent SDK.

This replaces the "spin up Claude Code interactively and prompt it" workflow from
the README with a single self-contained Python orchestrator. It drives the exact
experiment loop defined in program.md:

    edit train.py -> git commit -> train for 5 min -> read val_bpb ->
    keep (advance branch) or revert (git reset) -> log to results.tsv -> repeat

The orchestration lives here in Python; the *reasoning* (what to change, whether a
result is worth keeping) is done by the agent. A single persistent ClaudeSDKClient
session is used so the agent remembers what it already tried across experiments.

Usage:
    conda activate autoresearch          # must be the env that has torch etc.
    python main.py                       # ~100 experiments on a fresh branch
    python main.py --experiments 30
    python main.py --tag jun21 --model claude-opus-4-6
    python main.py --no-setup            # skip setup, continue an existing branch
"""

import argparse
import asyncio
import datetime
import re
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

# ---------------------------------------------------------------------------
# Enforcement layer (can_use_tool): the HARD guarantee, not the soft prompt.
#
# The SDK calls guard() before every tool use. We deny anything outside the
# training-code workspace. This is enforcement, independent of what the model
# "intends" or what the system prompt says.
# ---------------------------------------------------------------------------

# Files the agent is allowed to create/modify. train.py is the code space;
# results.tsv is the experiment log. Nothing else may be written.
WRITABLE_FILES = {"train.py", "results.tsv"}

# Bash commands matching any of these are hard-denied. The first group is the
# important one: anything that talks to a git remote or GitHub.
BASH_DENY_PATTERNS = [
    r"\bgit\s+push\b",
    r"\bgit\s+remote\b",
    r"\bgit\s+fetch\b",
    r"\bgit\s+pull\b",
    r"\bgit\s+clone\b",
    r"\bgit\s+(remote\s+)?(set-url|add)\b.*(https?://|git@)",
    r"\bgh\b",                       # GitHub CLI
    r"\bsudo\b",
    r"\brm\s+-rf\s+/(?!home/ruiyi/autoresearch)",  # rm -rf of anything but this repo
    r":\(\)\s*\{",                   # fork-bomb-ish
    r"\bcurl\b.*\|\s*(ba)?sh\b",     # curl | sh
    r"\bwget\b.*\|\s*(ba)?sh\b",
]


def _path_in_repo(file_path: str) -> bool:
    try:
        Path(file_path).resolve().relative_to(REPO)
        return True
    except (ValueError, RuntimeError):
        return False


async def guard(tool_name, tool_input, context):
    """Return Allow/Deny for each tool call. Deny = the agent cannot do it."""
    # 1) File writes: only train.py / results.tsv, only inside the repo.
    if tool_name in ("Write", "Edit", "MultiEdit", "NotebookEdit"):
        fp = tool_input.get("file_path", "")
        if not _path_in_repo(fp) or Path(fp).name not in WRITABLE_FILES:
            return PermissionResultDeny(
                message=f"Blocked: may only edit {sorted(WRITABLE_FILES)} inside {REPO}. "
                        f"prepare.py and all other files are read-only.",
            )
        return PermissionResultAllow()

    # 2) Bash: hard-deny remote git ops and other dangerous patterns.
    if tool_name == "Bash":
        cmd = tool_input.get("command", "")
        for pat in BASH_DENY_PATTERNS:
            if re.search(pat, cmd):
                return PermissionResultDeny(
                    message=f"Blocked by safety policy (matched /{pat}/). Remote git "
                            f"operations and out-of-repo destructive commands are not allowed.",
                )
        return PermissionResultAllow()

    # 3) Read-only / housekeeping tools (Read, Grep, Glob, TodoWrite): allow.
    return PermissionResultAllow()

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

# Appended to Claude Code's own system prompt so the agent keeps full tool
# behavior but learns the rules of THIS run. Two things differ from program.md:
# we run under conda (so `python`, not `uv run`), and training is launched in the
# background and polled so it can't hit the Bash tool's wall-clock timeout.
SYSTEM_APPEND = """
You are an autonomous ML research agent operating the `autoresearch` repo. Your
single objective is to minimize val_bpb (validation bits per byte) on the fixed
5-minute training budget. Follow program.md as the source of truth for the
workflow, EXCEPT for the two environment overrides below.

ENVIRONMENT OVERRIDES (this machine, not the defaults in program.md/README):
- We run inside a conda env, NOT uv. Whenever program.md says `uv run train.py`
  or `uv run prepare.py`, run `python train.py` / `python prepare.py` instead.
- The GPU is an A100 (not H100), so the mfu_percent number in the output is
  computed against the wrong peak FLOPS — ignore it. val_bpb is still valid and
  is the only metric that matters.

RUNNING A TRAINING EXPERIMENT (run it in the FOREGROUND, one blocking call):
- This is a SINGLE GPU and you do exactly ONE training run per turn. Run training
  as ONE foreground Bash command and WAIT for it to finish IN THE SAME TURN.
- Do NOT use `&` and do NOT use run_in_background. A backgrounded run makes your
  turn end before training completes, which breaks the whole loop (the orchestrator
  will think the experiment finished and move on while nothing actually ran).
- A run takes ~5 min of training + compile/eval overhead (~6-8 min total), so set
  the Bash tool `timeout` to its maximum (600000 ms) on that one command:
    python train.py > run.log 2>&1
- When it returns, read ONLY the metric lines (never echo the whole log):
    grep "^val_bpb:\\|^peak_vram_mb:" run.log
- If that grep is empty the run crashed: `tail -n 30 run.log`, then decide to fix
  (only if it's a trivial bug) or skip it as a crash and revert.
- If a single run ever exceeds the 600000 ms Bash timeout, treat it as a failure,
  revert, and move on — do not retry it in the background.

DISCIPLINE:
- One experiment = one idea. Make a focused change, commit, run, evaluate, decide.
- Keep (advance the branch) only if val_bpb strictly improved; otherwise
  `git reset --hard` back to the prior commit. Honor the simplicity criterion.
- Append every experiment to results.tsv (tab-separated). Do NOT git-commit
  results.tsv (it stays untracked).
- Do not modify prepare.py or the evaluate_bpb metric.
""".strip()

SETUP_PROMPT = """
Let's set up a new autonomous research run. Do the Setup section of program.md,
adapted to this environment:

1. Read README.md, program.md, prepare.py, and train.py for full context.
2. Create a fresh branch `autoresearch/{tag}` from the current branch
   (`git checkout -b autoresearch/{tag}`). If it already exists, stop and tell me.
3. Verify the data + tokenizer cache exists at ~/.cache/autoresearch/ (data
   shards + tokenizer). If it is missing, STOP and tell me to run
   `python prepare.py` first — do not try to run it yourself.
4. Initialize results.tsv with exactly this header row (tab-separated) and
   nothing else:  commit\tval_bpb\tmemory_gb\tstatus\tdescription
   Leave results.tsv untracked by git.

Then briefly confirm the setup looks good. Do NOT start training yet.
""".strip()

# The first real experiment must be the baseline (train.py unchanged).
FIRST_EXPERIMENT_PROMPT = """
Run experiment #1: the BASELINE. Do not change train.py at all. Commit the
current state if needed, run the training experiment (foreground, one blocking
call, as instructed), read out val_bpb and peak_vram_mb, and record the result in
results.tsv with status `keep` and description `baseline`. This establishes the
number every later experiment is compared against. Report the baseline val_bpb.
""".strip()

NEXT_EXPERIMENT_PROMPT = """
Run experiment #{n}. Pick ONE concrete, well-motivated change to train.py
(architecture, optimizer, hyperparameters, batch size, schedule, etc.) that you
believe will lower val_bpb relative to the best result so far. Briefly state your
hypothesis, edit train.py, git commit, run the experiment (foreground, one
blocking call), read val_bpb + peak_vram_mb, and:
  - if val_bpb strictly improved over the current best: keep it (branch advances),
    log status `keep`.
  - otherwise: `git reset --hard` to the prior commit, log status `discard`
    (or `crash` if it failed), and move on.
Append the row to results.tsv. Then give me a one-line summary: the change, the
val_bpb, and the keep/discard decision. Do not stop to ask whether to continue.
""".strip()


# ---------------------------------------------------------------------------
# Pretty-printing of the streamed agent messages
# ---------------------------------------------------------------------------

def _short(text, limit=2000):
    text = text.strip()
    return text if len(text) <= limit else text[:limit] + " […]"


async def drive_turn(client: ClaudeSDKClient, prompt: str):
    """Send one prompt and stream the agent's work until the turn completes."""
    await client.query(prompt)
    async for message in client.receive_response():
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    if block.text.strip():
                        print(f"\n🤖 {_short(block.text)}")
                elif isinstance(block, ThinkingBlock):
                    # Keep thinking quiet but show it's happening.
                    print("   …thinking…", flush=True)
                elif isinstance(block, ToolUseBlock):
                    arg = ""
                    if block.name == "Bash":
                        arg = block.input.get("command", "")
                    elif block.name in ("Edit", "Write", "Read"):
                        arg = block.input.get("file_path", "")
                    print(f"   ⚙️  {block.name}: {_short(str(arg), 200)}")
        elif isinstance(message, ResultMessage):
            cost = getattr(message, "total_cost_usd", None)
            turns = getattr(message, "num_turns", None)
            extra = []
            if turns is not None:
                extra.append(f"{turns} turns")
            if cost is not None:
                extra.append(f"${cost:.3f}")
            print(f"   ── turn complete ({', '.join(extra)}) ──")
            return message
    return None


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

async def run(args):
    options = ClaudeAgentOptions(
        cwd=str(REPO),
        model=args.model,
        system_prompt={
            "type": "preset",
            "preset": "claude_code",
            "append": SYSTEM_APPEND,
        },
        allowed_tools=["Read", "Write", "Edit", "Bash", "Grep", "Glob", "TodoWrite"],
        # Enforcement layer: guard() is consulted before EVERY tool call and is
        # what actually keeps the agent inside the training-code workspace.
        can_use_tool=guard,
        permission_mode="default",
        # One experiment needs many tool calls (edit, commit, launch, ~10 polls,
        # grep, log, decide). Give each turn plenty of headroom.
        max_turns=args.max_turns,
        # Let the SDK load CLAUDE.md / project settings if present.
        setting_sources=["project"],
    )

    print(f"Repo:        {REPO}")
    print(f"Model:       {args.model}")
    print(f"Branch tag:  autoresearch/{args.tag}")
    print(f"Experiments: {args.experiments}")
    print("=" * 70)

    results_tsv = REPO / "results.tsv"

    def result_rows():
        # number of logged experiments (data rows, excluding the header)
        if not results_tsv.exists():
            return 0
        return max(0, sum(1 for _ in results_tsv.open()) - 1)

    async with ClaudeSDKClient(options=options) as client:
        if not args.no_setup:
            print("\n### SETUP ###")
            await drive_turn(client, SETUP_PROMPT.format(tag=args.tag))

        for n in range(1, args.experiments + 1):
            print(f"\n### EXPERIMENT {n}/{args.experiments} ###")
            if n == 1 and not args.no_setup:
                prompt = FIRST_EXPERIMENT_PROMPT
            else:
                prompt = NEXT_EXPERIMENT_PROMPT.format(n=n)
            before = result_rows()
            try:
                await drive_turn(client, prompt)
            except Exception as e:  # keep the loop alive across transient failures
                print(f"   !! turn error: {e!r} — continuing to next experiment")
            # A real experiment appends exactly one row to results.tsv. If it
            # didn't grow, the turn did no real work (e.g. it only polled/waited) —
            # surface that instead of silently printing the next banner.
            if result_rows() == before:
                print(f"   ⚠️  no new results.tsv row after experiment {n} — "
                      f"the turn logged nothing (no real run completed).")

    print("\n" + "=" * 70)
    print("Done. See results.tsv for the experiment log and `git log` for kept changes.")


class _Tee:
    """Duplicate a text stream to the console AND a log file."""

    def __init__(self, stream, fh):
        self._stream = stream
        self._fh = fh

    def write(self, data):
        self._stream.write(data)
        self._fh.write(data)
        self._fh.flush()  # flush so the log is tail-able live
        return len(data)

    def flush(self):
        self._stream.flush()
        self._fh.flush()

    def __getattr__(self, name):  # delegate isatty(), encoding, etc.
        return getattr(self._stream, name)


def setup_logging(log_path: Path):
    """Mirror all stdout/stderr of this orchestrator into log_path (append mode)."""
    fh = open(log_path, "a", buffering=1, encoding="utf-8")
    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    fh.write(f"\n{'='*70}\n# main.py session started {stamp}\n{'='*70}\n")
    sys.stdout = _Tee(sys.stdout, fh)
    sys.stderr = _Tee(sys.stderr, fh)
    return fh


def main():
    default_tag = datetime.datetime.now().strftime("%b%d").lower()  # e.g. 'jun21'
    p = argparse.ArgumentParser(description="Autonomous autoresearch loop via Claude Agent SDK")
    p.add_argument("--experiments", type=int, default=100, help="number of experiments to run")
    p.add_argument("--tag", default=default_tag, help="run tag -> branch autoresearch/<tag>")
    p.add_argument("--model", default="claude-opus-4-8", help="model id for the agent")
    p.add_argument("--max-turns", type=int, default=80, help="max agent turns per experiment")
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
