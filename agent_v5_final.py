"""v5 - The finished agent beside the baseline.

This file adds no new pattern. It runs the v0 baseline and the v4 agent (all four
patterns) on the same history and request, then prints what each one cost and whether
its patch fixes the failing test.

The rule the patterns share: the signature states the question, Jev returns the
probability, and plain Python owns the policy.
"""

import subprocess
import time

import dspy

import config
from agent_v0_baseline import (
    HISTORY, REPO_DIR, USER_REQUEST, baseline_agent, reset_sample_repo, show_inputs,
)
from agent_v0_baseline import run_shell_command as run_ungated_command
from agent_v4_delegation import DelegatingCodingAgent

FAILING_TEST = "tests/test_parse_config.py::test_parses_nested_sections"


def count_tokens(entries: list[dict], *names: str) -> int:
    """Sum a token count across usage entries; providers name the fields differently."""
    return sum(next((entry[name] for name in names if entry.get(name) is not None), 0) for entry in entries)


def patch_fixes_test(patch: str) -> str:
    """Apply the agent's patch to a fresh copy of the repo and run the failing test."""
    reset_sample_repo()
    # --batch: when the diff's paths don't match -p1, patch otherwise prompts
    # "File to patch:" on /dev/tty (not stdin) and waits forever.
    applied = subprocess.run(["patch", "-p1", "--forward", "--batch"], input=patch, cwd=REPO_DIR,
                             capture_output=True, text=True, timeout=60)
    if applied.returncode != 0:
        return "patch did not apply"
    passed = run_ungated_command(f"pytest -q {FAILING_TEST}").startswith("exit code 0")
    return "yes" if passed else "no"


def measure(name: str, agent) -> dict:
    """Run one agent from a fresh repo and collect its usage from DSPy's tracker."""
    print(f"\n=== {name} ===")
    reset_sample_repo()
    start = time.perf_counter()
    with dspy.track_usage() as usage:  # records one entry per LM call, keyed by model
        result = agent(history=HISTORY, user_request=USER_REQUEST)
    seconds = time.perf_counter() - start
    frontier = usage.usage_data.get(config.frontier.model, [])
    return {
        "agent": name,
        "frontier in": count_tokens(frontier, "prompt_tokens", "input_tokens"),
        "frontier out": count_tokens(frontier, "completion_tokens", "output_tokens"),
        "jev calls": len(usage.usage_data.get(config.jev.model, [])),
        "sub_lm calls": len(usage.usage_data.get(config.cheap.model, [])),
        "seconds": round(seconds),
        "test passes": patch_fixes_test(result.patch),
    }


if __name__ == "__main__":
    # Turn off DSPy's request cache so both runs make, and count, every call.
    dspy.configure_cache(enable_disk_cache=False, enable_memory_cache=False)
    show_inputs()
    print("Running both agents with the cache off (this takes a few minutes)...")
    rows = [
        measure("v0 baseline", baseline_agent),
        measure("v4 all patterns", DelegatingCodingAgent()),
    ]
    columns = list(rows[0])
    widths = [max(len(column), *(len(str(row[column])) for row in rows)) for column in columns]
    print("\n" + " | ".join(c.ljust(w) for c, w in zip(columns, widths)))
    print("-+-".join("-" * w for w in widths))
    for row in rows:
        print(" | ".join(str(row[c]).ljust(w) for c, w in zip(columns, widths)))
