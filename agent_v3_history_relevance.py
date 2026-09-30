"""v3 - Pattern 3: score history for relevance before each turn.

Flaw it fixes: v2 re-reads its own history every turn to rediscover what it already
learned.

Before the agent starts, Jev scores every past turn against the current request on a
three-level rubric: hide, summarize, or show. Python keeps the "show" chunks verbatim,
has the cheap model summarize the "summarize" chunks, and drops the rest from view.

This is the first pattern that changes what the model *sees* rather than what it can
*do*, so it runs outside the sandbox, before the RLM starts. It replaces compaction
(summarize everything once) with compression aimed at the current request.

The rule: the signature states the question, Jev returns the probability, and plain
Python owns the policy.
"""

import dspy
from dspy.experimental import Score

from agent_v0_baseline import HISTORY, MAX_ITERATIONS, USER_REQUEST, reset_sample_repo, show_inputs
from agent_v2_find_relevant_tools import ALL_TOOLS, TOOL_FINDING_SIGNATURE, run_tests
from config import cheap, jev

# Score takes level descriptions in increasing order. Jev returns a probability per
# level; `.value` is the probability-weighted mean level, from 0 (hide) to 2 (show).
RELEVANCE_LEVELS = ["hide", "summarize", "show"]
Relevance = Score[
    "Nothing here helps with the request.",
    "The gist matters; the detail does not.",
    "The model needs this verbatim.",
]

# Python turns the mean level into an action with two named cut points.
SUMMARIZE_AT = 0.5
SHOW_AT = 1.5


class RankHistoryChunkRelevance(dspy.Signature):
    """Decide how much of one past turn the model should see for the current request."""

    user_request: str = dspy.InputField()
    chunk: str = dspy.InputField(desc="One past turn from the agent's history.")
    relevance: Relevance = dspy.OutputField(desc="How much of this chunk should the model see?")


CONTEXT_INSTRUCTIONS = (
    "\n`relevant_context` holds the past turns that matter for this request, verbatim. "
    "`context_summaries` holds summaries of turns that partly matter. Start from those; "
    "search the full `history` only if you need something they lack."
)


class HistoryAwareCodingAgent(dspy.Module):
    def __init__(self, tools=ALL_TOOLS, outputs="patch: str", instructions=None, max_iters=MAX_ITERATIONS):
        super().__init__()
        self.rank_chunk = dspy.Predict(RankHistoryChunkRelevance)
        self.summarize = dspy.Predict("chunk -> summary")
        instructions = (instructions or TOOL_FINDING_SIGNATURE.instructions) + CONTEXT_INSTRUCTIONS
        self.agent = dspy.RLM(
            dspy.Signature(
                "history: list[str], user_request: str, "
                f"relevant_context: list[str], context_summaries: list[str] -> {outputs}",
                instructions,
            ),
            tools=tools,
            sub_lm=cheap,
            max_iters=max_iters,
        )

    def select_context(self, history: list[str], user_request: str) -> tuple[list[str], list[str]]:
        """Score every chunk with Jev; return (chunks to show, summaries of chunks to summarize)."""
        relevant_context, context_summaries = [], []
        # A simple loop is fine here. Jev is priced on input tokens only, and these calls
        # are independent, so they could also be batched with self.rank_chunk.batch(...).
        for chunk in history:
            with dspy.context(lm=jev):
                relevance = self.rank_chunk(user_request=user_request, chunk=chunk).relevance
            level = "show" if relevance.value >= SHOW_AT else "summarize" if relevance.value >= SUMMARIZE_AT else "hide"
            probabilities = " ".join(
                f"{name}={relevance.probabilities[i]:.2f}" for i, name in enumerate(RELEVANCE_LEVELS)
            )
            print(f"    [history] {level:9} ({probabilities})  {chunk.splitlines()[0][:60]}")
            if level == "show":
                relevant_context.append(chunk)
            elif level == "summarize":
                with dspy.context(lm=cheap):
                    context_summaries.append(self.summarize(chunk=chunk).summary)
        return relevant_context, context_summaries

    def forward(self, history: list[str], user_request: str):
        relevant_context, context_summaries = self.select_context(history, user_request)
        # The full history still goes in, so the model can search it if Jev hid
        # something it turns out to need.
        return self.agent(
            history=history,
            user_request=user_request,
            relevant_context=relevant_context,
            context_summaries=context_summaries,
        )


if __name__ == "__main__":
    show_inputs()
    agent = HistoryAwareCodingAgent()
    print("Jev scores each past turn against the request:")
    relevant_context, context_summaries = agent.select_context(HISTORY, USER_REQUEST)
    print(f"\nrelevant_context: {len(relevant_context)} chunks, {sum(map(len, relevant_context))} chars")
    print(f"context_summaries: {len(context_summaries)} summaries, {sum(map(len, context_summaries))} chars")
    print(f"full history: {len(HISTORY)} chunks, {sum(map(len, HISTORY))} chars")

    print("\nRunning the history-aware agent:")
    reset_sample_repo()
    result = agent(history=HISTORY, user_request=USER_REQUEST)
    print("PATCH:\n", result.patch)
    print(f"\nIterations used: {len(result.trajectory)} of {MAX_ITERATIONS}")
    print("Tests afterward:", run_tests().splitlines()[-1])

# The flaw: the agent does everything in sequence, even steps that don't depend on each
# other. Pattern 4 (agent_v4) notices work a read-only subagent could take.
