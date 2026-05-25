"""LLM-as-calculator: send a math expression to OpenAI and print the result.

Usage:
    python calculator.py "2 + 2 * 10"
    echo "sqrt(144)" | python calculator.py
"""

from __future__ import annotations

import sys

from dotenv import load_dotenv
from openai import OpenAI, OpenAIError

MODEL = "gpt-4o-mini"

# Sentinel the model returns for anything that is not a classic calculator input.
REFUSAL = "NOT_A_CALCULATION"

SYSTEM_PROMPT = f"""You are a calculator. You evaluate mathematical expressions \
and nothing else.

A "classic calculator input" is a mathematical expression made of numbers and \
the operations a standard scientific calculator supports: addition, subtraction, \
multiplication, division, exponentiation, modulo, parentheses, and common \
functions such as sqrt, sin, cos, tan, log, ln, abs, and constants like pi and e.

Rules:
- If the input is a valid mathematical expression, reply with ONLY the numeric \
result. No words, no units, no explanation, no trailing punctuation.
- Round non-integer results to at most 10 significant digits.
- If the input is NOT a pure mathematical calculation (e.g. it asks a question, \
requests text, contains words that are not math functions, or is otherwise not \
something a physical calculator could compute), reply with exactly: {REFUSAL}

Examples:
- "2 + 2" -> "4"
- "(3 + 4) * 2" -> "14"
- "sqrt(144)" -> "12"
- "sin(0)" -> "0"
- "10 / 4" -> "2.5"
- "what is the capital of France" -> "{REFUSAL}"
- "write me a poem" -> "{REFUSAL}"
- "2 + 2 then tell me a joke" -> "{REFUSAL}"
"""


class RefusalError(Exception):
    """Raised when the input is not a classic calculator input."""


def calculate(expression: str, client: OpenAI | None = None) -> str:
    """Evaluate ``expression`` via the LLM and return the result string.

    Raises ``RefusalError`` if the input is not a valid calculator expression.
    """
    if not expression or not expression.strip():
        raise RefusalError("empty input")

    client = client or OpenAI()
    response = client.chat.completions.create(
        model=MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": expression.strip()},
        ],
    )
    result = (response.choices[0].message.content or "").strip()

    if result == REFUSAL or not result:
        raise RefusalError(
            f"refused: {expression.strip()!r} is not a classic calculator input"
        )
    return result


def main() -> int:
    load_dotenv()

    if len(sys.argv) > 1:
        expression = " ".join(sys.argv[1:])
    else:
        expression = sys.stdin.read()

    try:
        print(calculate(expression))
        return 0
    except RefusalError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except OpenAIError as exc:
        print(f"API error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
