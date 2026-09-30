"""One place to configure every model and the sandbox the series uses.

Three models play three roles:
  - `frontier` writes code. DSPy uses it by default for every module.
  - `cheap` is the RLM's `sub_lm`: the sandbox calls it through `llm_query(...)` to read
    slices of text, and Pattern 3 uses it to summarize history chunks.
  - `jev` is TypeSafe's decision model. It answers closed questions with probabilities.
    We never configure it globally; each decision wraps its call in
    `with dspy.context(lm=jev):`.

Change a model here and every file in the series picks it up.
"""

from pathlib import Path

import dspy
from dotenv import load_dotenv
from dspy.experimental import TypeSafe
from dspy_monty_interpreter import MontyInterpreter

# load_dotenv reads OPENAI_API_KEY and TYPESAFE_API_KEY from the .env file next to this one.
load_dotenv(Path(__file__).parent / ".env")

# The frontier model writes the sandbox code. We use an OpenAI model here because Claude
# Sonnet 5.5 and Opus 5.5 refused dspy.RLM's action prompt during testing.
frontier = dspy.LM("openai/gpt-6-luna", timeout=120)  # timeout: retry a stalled request instead of hanging

# The cheap model reads slices and writes summaries.
cheap = dspy.LM("openai/gpt-6-sol", timeout=120)  # timeout: retry a stalled request instead of hanging

# Jev reads TYPESAFE_API_KEY from the environment.
jev = TypeSafe("jev-1.13.0")

# Monty is Pydantic's Python interpreter, written in Rust. We use it for two reasons:
#   1. It starts in microseconds. DSPy creates a fresh interpreter on every RLM
#      forward() call, so startup cost adds up.
#   2. It has no filesystem, network, or environment access by default. Sandbox code
#      can reach the outside world only through the tools we register, which makes
#      the permission gate in Pattern 1 the only door.
# We deliberately do not mount sample_repo into Monty; every file read goes through a
# tool so it shows up in the trajectory.
dspy.configure(
    lm=frontier,
    interpreter_factory=MontyInterpreter.factory(request_timeout=30.0),
)
