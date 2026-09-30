# A coding agent built on DSPy's RLM and TypeSafe's Jev

Six small files build one coding agent, and each file adds a single design pattern to the one before it. Every pattern follows the same rule: **the signature states the question, Jev returns the probability, and plain Python owns the policy.**

The agent's large model (the "frontier" model) writes code inside a `dspy.RLM` sandbox. Jev, TypeSafe's decision model, answers closed questions with probabilities: is this command safe, which tool fits, how much of this past turn matters, and could a helper take this step. Plain Python turns those probabilities into actions using named thresholds.

## Setup

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python "dspy[typesafe]==3.4.0" "dspy-monty-interpreter>=0.5.0" pytest python-dotenv
```

Put `OPENAI_API_KEY` and `TYPESAFE_API_KEY` in `.env`. To change a model, edit `config.py`; every file picks up the change.

Each demo resets `sample_repo/` from `sample_repo_pristine/` before it runs. That repo has one bug, and `tests/test_parse_config.py::test_parses_nested_sections` fails because of it. Every demo sends the same request, along with the eight fake prior turns in `sample_history.py`.

## Run the series in order

```bash
.venv/bin/python agent_v0_baseline.py
.venv/bin/python agent_v1_permission_gate.py
.venv/bin/python agent_v2_find_relevant_tools.py
.venv/bin/python agent_v3_history_relevance.py
.venv/bin/python agent_v4_delegation.py
.venv/bin/python agent_v5_final.py
.venv/bin/python fit_thresholds.py
```

DSPy caches identical requests, so running a demo a second time replays the first run's answers in seconds. `agent_v5_final.py` turns the cache off so its numbers are real. Set `AUTO_APPROVE=yes` to approve commands that fall into the gate's "ask" band. The default is to deny them.

## The files

**`config.py`** sets up the three models in one place: the frontier model, a cheaper `sub_lm` for reading and summarizing, and Jev. It also installs Monty as the sandbox. Monty starts in microseconds and has no filesystem, network, or environment access, so the only way sandbox code can reach the outside world is through the tools we register.

**`agent_v0_baseline.py`** is a plain `dspy.RLM` with two tools, `run_shell_command` and `search_files`. The history arrives as a sandbox variable, so the model sees only a preview of it. Its flaw is that it runs any command it writes.

**`agent_v1_permission_gate.py`** (Pattern 1) puts a Jev yes-or-no question, `IsCommandSafeToRun`, inside `run_shell_command`. Below `DENY_BELOW` the command is refused, between the thresholds the user is asked, and above `ASK_BELOW` it runs. Because the gate lives inside the tool, nothing the model writes can get around it.

**`agent_v2_find_relevant_tools.py`** (Pattern 2) registers fourteen tools, each with a one-line docstring, plus `find_relevant_tools(goal)`. That function asks Jev one `Choice` question over every tool and returns the top k along with their full documentation. The model sees a ranked short list instead of every schema at once.

**`agent_v3_history_relevance.py`** (Pattern 3) adds `HistoryAwareCodingAgent`. Before the RLM starts, Jev scores every past turn as hide, summarize, or show (a `Score`). The "show" turns go in verbatim as `relevant_context`, and the cheap model's summaries of the "summarize" turns go in as `context_summaries`. This is the first pattern that changes what the model sees, not what it can do.

**`agent_v4_delegation.py`** (Pattern 4) runs the agent in short bursts because `dspy.RLM` has no per-iteration hook. After each burst, Jev answers three yes-or-no questions (`IsStepDelegatable`) about each remaining step: is it independent, is it read-only, and has it not been delegated yet. When all three reach `DELEGATE_AT`, Python starts a read-only subagent in a thread, and Pattern 3 picks that subagent's context against the subgoal. The subagent's report returns to the parent as a history chunk.

**`agent_v5_final.py`** runs v0 and v4 on the same input with the cache off. It prints frontier-model tokens, Jev calls, `sub_lm` calls, wall-clock time, and whether each agent's patch, applied to a fresh copy of the repo, fixes the failing test.

**`fit_thresholds.py`** fits the gate's run-versus-ask threshold using ReAnchor, 32 labeled commands, and a metric that treats a false approval as ten times worse than a false denial.
