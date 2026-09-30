"""v4 - Pattern 4: notice work a subagent could take.

Flaw it fixes: v3 does everything in sequence, even steps that don't depend on each other.

Agents rarely spawn helpers because two jobs fell on the expensive model: noticing the
chance, and choosing what context to hand over. Pattern 3 already handles the second.
This pattern handles the first: after every burst, Jev answers three yes-or-no questions
about each remaining step, and Python spawns a read-only subagent when all three clear.

The rule: the signature states the question, Jev returns the probability, and plain
Python owns the policy.
"""

import contextvars
from concurrent.futures import ThreadPoolExecutor

import dspy
from dspy.experimental import Noul

from agent_v0_baseline import HISTORY, USER_REQUEST, reset_sample_repo, show_inputs
from agent_v2_find_relevant_tools import (
    ALL_TOOLS, TOOL_FINDING_SIGNATURE, TOOL_REGISTRY, git_diff, run_tests,
)
from agent_v3_history_relevance import HistoryAwareCodingAgent
from config import jev

DELEGATE_AT = 0.7  # every one of the three probabilities must reach this to spawn
BURST_ITERATIONS = 4  # RLM iterations per burst; the wrapper checks between bursts
MAX_BURSTS = 8
READ_ONLY_TOOL_NAMES = [
    "search_files", "search_code_structure", "read_file", "list_directory", "git_diff",
    "git_log", "lint_file", "find_symbol_definition", "find_callers",
]


class IsStepDelegatable(dspy.Signature):
    """Decide whether a remaining step could be handed to a read-only subagent right now."""

    remaining_plan: str = dspy.InputField(desc="The remaining steps as the agent understands them.")
    trajectory_so_far: str = dspy.InputField()
    delegated_subgoals: list[str] = dspy.InputField(desc="Subgoals already handed to subagents.")
    current_step: str = dspy.InputField(desc="The step the agent is working on now.")
    candidate_step: str = dspy.InputField(desc="A later step from the remaining plan.")
    independent: Noul = dspy.OutputField(
        desc="This step would give the same result whether it runs before or after the current step.")
    read_only: Noul = dspy.OutputField(desc="This step only reads the codebase; it writes nothing.")
    not_yet_delegated: Noul = dspy.OutputField(desc="No delegated subgoal already covers this step.")


check_step = dspy.Predict(IsStepDelegatable)
REQUESTED_SUBGOALS: list[str] = []  # subgoals that cleared; the wrapper spawns them


def find_delegatable_step(remaining_plan: str, trajectory_so_far: str, delegated_subgoals: list[str]) -> dict:
    """Check whether the remaining work has a step a read-only subagent could take now."""
    # Python splits the plan into steps. The first step is the one the agent is on; Jev
    # scores each later step, so "this step" in every question means the same step.
    # Jev answers only closed questions, so the subgoal is the step's own text rather
    # than a generated sentence.
    steps = [line.strip("-*0123456789. ") for line in remaining_plan.splitlines() if line.strip()]
    for candidate in steps[1:]:
        with dspy.context(lm=jev):
            answer = check_step(
                remaining_plan=remaining_plan, trajectory_so_far=trajectory_so_far[-6000:],
                delegated_subgoals=delegated_subgoals, current_step=steps[0], candidate_step=candidate,
            )
        probabilities = {
            "independent": answer.independent.probability,
            "read_only": answer.read_only.probability,
            "not_yet_delegated": answer.not_yet_delegated.probability,
        }
        delegate = all(p >= DELEGATE_AT for p in probabilities.values())
        print("    [delegate] " + " ".join(f"{k}={p:.2f}" for k, p in probabilities.items())
              + f" {'SPAWN' if delegate else 'keep '}  {candidate}")
        if delegate:
            REQUESTED_SUBGOALS.append(candidate)
            return {**probabilities, "delegate": True, "subgoal": candidate}
    return {"delegate": False, "subgoal": None}


PLAN_INSTRUCTIONS = (
    "\nYou work in short bursts. Set `remaining_plan` to the steps you still intend to take, "
    "one per line. Include read-only checks, such as finding other callers of the code you "
    "change. "
    "Set `done` to true only when the fix is made and the tests pass. All bursts belong to "
    "one run: earlier bursts appear in `relevant_context` as '[burst N]' chunks and subagent "
    "reports as '[subagent: ...]' chunks. Print those chunks first, pick up from their "
    "remaining plan, and do not repeat tool lookups or file reads they already contain. "
    "You may call find_delegatable_step(...) when a read-only step could run in parallel."
)

SUBAGENT_INSTRUCTIONS = (
    "You are a read-only helper. Investigate the goal in `user_request` with your tools "
    "and report what you found in `findings`. You cannot edit files."
)


def summarize_trajectory(trajectory: list[dict]) -> str:
    """Condense RLM steps into text the next burst can read from `history`."""
    return "\n".join(f"code:\n{step['code']}\noutput:\n{step['output'][:1500]}" for step in trajectory)


THIS_RUN_PREFIXES = ("[burst", "[subagent:")


class BurstAwareCodingAgent(HistoryAwareCodingAgent):
    """A HistoryAwareCodingAgent that always shows chunks from the current run."""

    def select_context(self, history, user_request):
        # Jev scores only the earlier history. This run's bursts and subagent reports are
        # the agent's working memory, so Python always shows them verbatim.
        this_run = [chunk for chunk in history if chunk.startswith(THIS_RUN_PREFIXES)]
        earlier = [chunk for chunk in history if not chunk.startswith(THIS_RUN_PREFIXES)]
        relevant_context, context_summaries = super().select_context(earlier, user_request)
        return relevant_context + this_run, context_summaries


class DelegatingCodingAgent(dspy.Module):
    def __init__(self):
        super().__init__()
        self.worker = BurstAwareCodingAgent(
            tools=[*ALL_TOOLS, find_delegatable_step],
            outputs="patch: str, remaining_plan: str, done: bool",
            instructions=TOOL_FINDING_SIGNATURE.instructions + PLAN_INSTRUCTIONS,
            max_iters=BURST_ITERATIONS,
        )
        # Subagents get read-only tools only. Writers would need file locks, which this
        # demo does not implement, so we never delegate a step that writes.
        self.subagent = HistoryAwareCodingAgent(
            tools=[TOOL_REGISTRY[name] for name in READ_ONLY_TOOL_NAMES],
            outputs="findings: str", instructions=SUBAGENT_INSTRUCTIONS, max_iters=6,
        )

    def run_subagent(self, subgoal: str, history: list[str]) -> str:
        # Pattern 3 scores history against the subgoal, because it arrives as user_request.
        return self.subagent(history=history, user_request=subgoal).findings

    def forward(self, history: list[str], user_request: str):
        history, delegated, running = list(history), [], {}
        # dspy.RLM has no per-iteration hook, so this wrapper runs the RLM in short bursts
        # and checks for delegatable work between them. Each burst's trajectory goes back
        # in through `history`, because the sandbox does not survive between calls.
        with ThreadPoolExecutor() as pool:
            for burst in range(1, MAX_BURSTS + 1):
                result = self.worker(history=history, user_request=user_request)
                trajectory = summarize_trajectory(result.trajectory)
                history.append(f"[burst {burst}] remaining plan: {result.remaining_plan}\n{trajectory}")
                print(f"  burst {burst}: {len(result.trajectory)} steps, done={result.done}")
                print("    remaining plan: " + result.remaining_plan.replace("\n", "\n                    "))

                for subgoal, future in list(running.items()):
                    if future.done():
                        print(f"  subagent returned for {subgoal!r}:\n    {future.result()[:300]}")
                        history.append(f"[subagent: {subgoal}] {future.result()}")
                        del running[subgoal]
                if result.done:
                    break

                # The wrapper checks after every burst, so noticing never depends on the
                # frontier model remembering to ask.
                trajectory_so_far = "\n".join(c for c in history if c.startswith(THIS_RUN_PREFIXES))
                find_delegatable_step(result.remaining_plan, trajectory_so_far, delegated)
                while REQUESTED_SUBGOALS:
                    subgoal = REQUESTED_SUBGOALS.pop(0)
                    print(f"  spawning read-only subagent: {subgoal!r}")
                    delegated.append(subgoal)
                    context = contextvars.copy_context()  # carries DSPy settings into the thread
                    running[subgoal] = pool.submit(context.run, self.run_subagent, subgoal, list(history))
        for subgoal, future in running.items():  # helpers that finished after the parent did
            print(f"  subagent returned for {subgoal!r} after the parent finished:\n    {future.result()[:300]}")
        return dspy.Prediction(patch=result.patch, bursts=burst, delegated_subgoals=delegated)


EXAMPLE_PLAN = """Change the section-header loop in parse_config.py to walk every dotted part
Find every other caller of parse_config in the repository
Run the full test suite"""
EXAMPLE_TRAJECTORY = "Read parse_config.py. The loop over section parts starts at parts[2:]."

if __name__ == "__main__":
    show_inputs()
    agent = DelegatingCodingAgent()
    print("Delegation check on an example plan:")
    check = find_delegatable_step(EXAMPLE_PLAN, EXAMPLE_TRAJECTORY, delegated_subgoals=[])
    REQUESTED_SUBGOALS.clear()
    if check["delegate"]:
        print(f"  spawning read-only subagent: {check['subgoal']!r}")
        print("  subagent returned:", agent.run_subagent(check["subgoal"], HISTORY)[:500])

    print("\nRunning the delegating agent:")
    reset_sample_repo()
    result = agent(history=HISTORY, user_request=USER_REQUEST)
    print("\nPATCH:\n", result.patch or git_diff())
    print(f"Bursts: {result.bursts}; delegated: {result.delegated_subgoals}")
    print("Tests afterward:", run_tests().splitlines()[-1])
