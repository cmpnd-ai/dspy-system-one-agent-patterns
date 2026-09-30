"""v0 - Baseline: a plain dspy.RLM coding agent.

Pattern: none yet. This is the agent every later file improves.
Flaw it fixes: none; it sets up the flaws the patterns fix.

The rule every later pattern follows: the signature states the question, Jev returns
the probability, and plain Python owns the policy. This file has no Jev yet.

How dspy.RLM works, briefly: each input field becomes a variable in a Python sandbox.
The frontier model sees only each variable's type, length, and a short preview, then
writes Python to read the parts it needs. The sandbox code can call our tools and
`llm_query(...)`, which hands a slice of text to the cheaper `sub_lm`. The model loops
until it calls SUBMIT(...) with the outputs, or until it hits the iteration limit.
"""

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import dspy

from config import cheap
from sample_history import HISTORY

USER_REQUEST = "The test test_parses_nested_sections is failing. Find the cause and fix it."

REPO_DIR = Path(__file__).parent / "sample_repo"
PRISTINE_REPO_DIR = Path(__file__).parent / "sample_repo_pristine"
MAX_ITERATIONS = 15


def reset_sample_repo() -> None:
    """Restore sample_repo from the pristine copy, so every demo starts with the bug."""
    shutil.rmtree(REPO_DIR, ignore_errors=True)
    shutil.copytree(PRISTINE_REPO_DIR, REPO_DIR)


def run_shell_command(cmd: str) -> str:
    """Run a shell command in the repository and return its output."""
    # Put this venv's bin directory first on PATH so `pytest` resolves to the installed one.
    env = {**os.environ, "PATH": f"{Path(sys.executable).parent}{os.pathsep}{os.environ['PATH']}"}
    completed = subprocess.run(
        cmd, shell=True, cwd=REPO_DIR, env=env, capture_output=True, text=True, timeout=120
    )
    return f"exit code {completed.returncode}\n{completed.stdout}{completed.stderr}"[-4000:]


def search_files(pattern: str, path: str = ".") -> str:
    """Search files under the repository for a regex; return matching lines as path:line: text."""
    root = (REPO_DIR / path).resolve()
    if not root.is_relative_to(REPO_DIR.resolve()):
        return f"error: {path} is outside the repository"
    regex = re.compile(pattern)
    matches = []
    files = [root] if root.is_file() else sorted(root.rglob("*"))
    for file in files:
        if not file.is_file() or "cache" in str(file):
            continue
        for number, line in enumerate(file.read_text(errors="ignore").splitlines(), start=1):
            if regex.search(line):
                matches.append(f"{file.relative_to(REPO_DIR)}:{number}: {line.strip()}")
    return "\n".join(matches[:100]) or "no matches"


def show_inputs(history: list[str] = HISTORY, user_request: str = USER_REQUEST) -> None:
    """Print what the agent receives: a one-line preview of each past turn, then the request."""
    print(f"=== HISTORY: {len(history)} past turns, {sum(map(len, history)):,} characters ===")
    for chunk in history:
        print(f"  {len(chunk):>5,} chars  {chunk.splitlines()[0][:70]}")
    print(f"\n=== USER REQUEST ===\n  {user_request}\n")


CODING_INSTRUCTIONS = """You are a coding agent working in a small Python repository.
Investigate with the tools, fix the problem by editing files in the repository, and run
the tests to confirm the fix. Submit `patch` as a unified diff of the change you made."""

# The signature has two inputs. `history` holds the earlier turns of this session. It
# arrives as a sandbox variable, so the model sees only a preview, and past turns cost
# nothing until the model's code touches them.
CODING_SIGNATURE = dspy.Signature(
    "history: list[str], user_request: str -> patch: str", CODING_INSTRUCTIONS
)

baseline_agent = dspy.RLM(
    CODING_SIGNATURE,
    tools=[run_shell_command, search_files],
    sub_lm=cheap,
    max_iters=MAX_ITERATIONS,
)


if __name__ == "__main__":
    show_inputs()
    print("Running the baseline agent (this takes about a minute)...")
    reset_sample_repo()
    result = baseline_agent(history=HISTORY, user_request=USER_REQUEST)
    print("PATCH:\n", result.patch)
    print(f"\nIterations used: {len(result.trajectory)} of {MAX_ITERATIONS}")
    print("Trajectory starts:", str(result.trajectory)[:200])
    print("\nTests afterward:", run_shell_command("pytest -q").splitlines()[-1])

# The flaw: this agent runs any command it writes. Nothing stands between the model's
# code and `rm -rf`, `curl | sh`, or `pip install`. Pattern 1 (agent_v1) adds a gate.
