"""v2 - Pattern 2: a tool that finds the right tools.

Flaw it fixes: v1 is safe enough to trust with many tools, but every tool's full schema
would sit in the prompt on every turn.

The agent now has fourteen tools, each registered with a one-line docstring. When it
needs one, it calls `find_relevant_tools(goal)`. Jev ranks every tool for that goal
with one Choice question, and Python returns the top k with their full documentation.

The rule: the signature states the question, Jev returns the probability, and plain
Python owns the policy.
"""

import difflib
import shlex

import dspy
from dspy.experimental import Choice

from agent_v0_baseline import (
    CODING_SIGNATURE, HISTORY, MAX_ITERATIONS, PRISTINE_REPO_DIR, REPO_DIR, USER_REQUEST,
    reset_sample_repo, search_files, show_inputs,
)
from agent_v1_permission_gate import run_shell_command
from config import cheap, jev


def _repo_path(path: str):
    resolved = (REPO_DIR / path).resolve()
    if not resolved.is_relative_to(REPO_DIR.resolve()):
        raise ValueError(f"{path} is outside the repository")
    return resolved


def _python_files():
    return [f for f in sorted(REPO_DIR.rglob("*.py")) if "cache" not in str(f)]


def search_code_structure(pattern: str) -> str:
    """List function and class definitions whose names match a regex."""
    return search_files(rf"^\s*(def|class)\s+\w*{pattern}")


def run_tests(test_path: str = "") -> str:
    """Run pytest, optionally on one test file or test id."""
    return run_shell_command(f"pytest -q {shlex.quote(test_path)}" if test_path else "pytest -q")


def read_file(path: str, start_line: int = 1, end_line: int = 200) -> str:
    """Read a range of numbered lines from a file."""
    lines = _repo_path(path).read_text().splitlines()
    return "\n".join(f"{n}: {line}" for n, line in enumerate(lines, 1) if start_line <= n <= end_line)


def list_directory(path: str = ".") -> str:
    """List the files and folders in a directory."""
    return "\n".join(p.name + ("/" if p.is_dir() else "") for p in sorted(_repo_path(path).iterdir()))


def git_diff() -> str:
    """Show a unified diff of every change made to the repository so far."""
    # sample_repo has no .git directory, so this compares against the pristine copy.
    diff = []
    for new in _python_files():
        old = PRISTINE_REPO_DIR / new.relative_to(REPO_DIR)
        old_lines = old.read_text().splitlines(True) if old.exists() else []
        name = str(new.relative_to(REPO_DIR))
        diff += difflib.unified_diff(old_lines, new.read_text().splitlines(True), f"a/{name}", f"b/{name}")
    return "".join(diff) or "no changes"


def git_log(n: int = 10) -> str:
    """Show recent commits."""
    # Honest stub: sample_repo is not a git repository.
    return "sample_repo has no git history; earlier turns in `history` include a git log."


def format_code(path: str) -> str:
    """Strip trailing whitespace and end the file with one newline."""
    file = _repo_path(path)
    file.write_text("\n".join(line.rstrip() for line in file.read_text().splitlines()) + "\n")
    return f"formatted {path}"


def lint_file(path: str) -> str:
    """Check a Python file for syntax errors."""
    try:
        compile(_repo_path(path).read_text(), path, "exec")
        return f"{path}: ok"
    except SyntaxError as error:
        return f"{path}:{error.lineno}: {error.msg}"


def find_symbol_definition(name: str) -> str:
    """Find where a function or class is defined."""
    return search_files(rf"^\s*(def|class)\s+{name}\b")


def find_callers(name: str) -> str:
    """Find every line that calls a function."""
    hits = search_files(rf"\b{name}\(").splitlines()
    return "\n".join(h for h in hits if f"def {name}" not in h) or "no callers"


def write_file(path: str, content: str) -> str:
    """Overwrite a file with new content."""
    _repo_path(path).write_text(content)
    return f"wrote {len(content)} characters to {path}"


def replace_in_file(path: str, old: str, new: str) -> str:
    """Replace one exact snippet of text in a file."""
    file = _repo_path(path)
    text = file.read_text()
    if text.count(old) != 1:
        return f"error: expected exactly one match, found {text.count(old)}"
    file.write_text(text.replace(old, new))
    return f"replaced 1 snippet in {path}"


TOOL_REGISTRY: dict[str, dspy.Tool] = {
    tool.name: tool
    for tool in map(dspy.Tool, [
        search_files, search_code_structure, run_tests, read_file, list_directory, git_diff,
        git_log, format_code, lint_file, find_symbol_definition, find_callers, write_file,
        replace_in_file, run_shell_command,
    ])
}

# The full documentation, which the agent sees only through find_relevant_tools.
TOOL_DOCS: dict[str, str] = {
    "search_files": "search_files(pattern, path='.') -> 'file:line: text' for each regex match under path.",
    "search_code_structure": "search_code_structure(pattern) -> def/class lines whose name contains the regex.",
    "run_tests": "run_tests(test_path='') -> pytest output. test_path may be a file or 'file::test_name'.",
    "read_file": "read_file(path, start_line=1, end_line=200) -> numbered lines. Paths are repo-relative.",
    "list_directory": "list_directory(path='.') -> one entry per line; folders end with '/'.",
    "git_diff": "git_diff() -> unified diff of all edits so far. Use it to build the final patch.",
    "git_log": "git_log(n=10) -> recent commits. sample_repo has none; check `history` instead.",
    "format_code": "format_code(path) -> strips trailing whitespace in place. Run after editing.",
    "lint_file": "lint_file(path) -> 'ok' or 'path:line: message' for the first syntax error.",
    "find_symbol_definition": "find_symbol_definition(name) -> the def/class line for an exact name.",
    "find_callers": "find_callers(name) -> every 'name(' call site, excluding the definition.",
    "write_file": "write_file(path, content) -> overwrites the whole file. Prefer replace_in_file for small edits.",
    "replace_in_file": "replace_in_file(path, old, new) -> replaces text that must match exactly once, "
                       "including indentation. Read the file first to copy `old` exactly.",
    "run_shell_command": "run_shell_command(cmd) -> exit code and output. Passes a permission gate; "
                         "may return 'denied: <cmd>'.",
}

# One Choice option per tool. Jev returns a probability for every option in one call.
RankedTool = Choice[tuple((t.name, t.desc) for t in TOOL_REGISTRY.values())]


class RankToolsForGoal(dspy.Signature):
    """Rank the available tools for what the agent is trying to do next."""

    goal: str = dspy.InputField(desc="What the agent is trying to do next, in one sentence.")
    best_tool: RankedTool = dspy.OutputField(desc="Which tool best advances this goal?")


rank_tools = dspy.Predict(RankToolsForGoal)


def find_relevant_tools(goal: str, k: int = 3) -> list[dict]:
    """Rank the available tools for a goal; return the top k with full argument schemas."""
    with dspy.context(lm=jev):
        probabilities = rank_tools(goal=goal).best_tool.probabilities
    ranked = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)[:k]
    print(f"    [tools] {goal!r} -> " + ", ".join(f"{name} {p:.2f}" for name, p in ranked))
    return [
        {"name": name, "probability": round(p, 3), "args": TOOL_REGISTRY[name].args, "doc": TOOL_DOCS[name]}
        for name, p in ranked
    ]


TOOL_INSTRUCTIONS = (
    "\nBefore using a tool for the first time in a run, call `find_relevant_tools(goal)` "
    "to get its argument schema."
)

# The RLM lists every tool by name and one-line docstring. The frontier model never sees
# every full schema at once; it sees a ranked short list, and it may reasonably pick the
# second option because its arguments fit the job better.
TOOL_FINDING_SIGNATURE = CODING_SIGNATURE.with_instructions(CODING_SIGNATURE.instructions + TOOL_INSTRUCTIONS)
ALL_TOOLS = [*TOOL_REGISTRY.values(), find_relevant_tools]

tool_finding_agent = dspy.RLM(TOOL_FINDING_SIGNATURE, tools=ALL_TOOLS, sub_lm=cheap, max_iters=MAX_ITERATIONS)


if __name__ == "__main__":
    show_inputs()
    for tool in find_relevant_tools("locate every caller of parse_config", k=3):
        print(f"  {tool['probability']:.2f}  {tool['doc']}")

    print("\nRunning the tool-finding agent:")
    reset_sample_repo()
    result = tool_finding_agent(history=HISTORY, user_request=USER_REQUEST)
    print("PATCH:\n", result.patch)
    print(f"\nIterations used: {len(result.trajectory)} of {MAX_ITERATIONS}")
    print("Tests afterward:", run_tests().splitlines()[-1])

# The flaw: every turn, the agent re-reads its own history to rediscover what it already
# learned. Pattern 3 (agent_v3) scores history before the turn starts.
