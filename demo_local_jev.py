"""Demo: swap the TypeSafe endpoint.

The same signature and Predict run against three endpoints: hosted Jev, a gateway, and a
local Tev1 served by Ollama. Only the TypeSafe constructor changes.
Local setup: `ollama pull tev1:0.8b`.
"""

import os

import dspy
from dotenv import load_dotenv
from dspy.experimental import Noul, TypeSafe

load_dotenv()

ENDPOINTS = {
    "typesafe": TypeSafe("jev-1.13.0"),  # reads TYPESAFE_API_KEY
    "lunaroute": TypeSafe(
        "djev",
        api_key=os.environ["LUNAROUTE_API_KEY"],
        base_url="https://gw.lunaroute.com/",
    ),
    "ollama": TypeSafe("tev1:0.8b", api_key="ollama", base_url="http://localhost:11434"),
}


class IsCommandSafeToRun(dspy.Signature):
    """Decide whether a shell command is safe to run without asking the user."""

    command: str = dspy.InputField()
    safe: Noul = dspy.OutputField(desc="This command is safe to run without asking the user.")


check_command = dspy.Predict(IsCommandSafeToRun)

for name, jev in ENDPOINTS.items():
    print(f"=== {name}: {jev.model} @ {jev.base_url} ===")
    for command in ["pytest -q", "rm -rf ~"]:
        try:
            with dspy.context(lm=jev):
                probability = check_command(command=command).safe.probability
            print(f"  P(safe)={probability:.2f}  {command}")
        except Exception as error:
            print(f"  error: {error}")
