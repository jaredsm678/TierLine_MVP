"""Prompt renderers for the two frozen prompt versions (design doc §6).

Frozen text rules (§6):
- V1 (canonical) and V2 (rewording) differ only in framing; the definitions
  block and the option order are byte-identical across versions and models.
- Single message, no system role — the LLM adapter sends the rendered prompt
  as one user message.
- The utterance is fenced in triple quotes, verbatim.

The definitions block is rendered as one bullet per option in the frozen
alphabetical order of schema.OPTIONS, using the single-logical-line
definition text from schema.DEFINITIONS (see the transcription notes there).
"""

from __future__ import annotations

from tierline_mvp import schema

PROMPT_VERSIONS = ("V1", "V2")


def render_definitions_block() -> str:
    """The §5 definitions block: '- name: definition', frozen option order."""
    return "\n".join(
        f"- {name}: {schema.DEFINITIONS[name]}" for name in schema.OPTIONS
    )


def render_prompt(version: str, utterance: str) -> str:
    """Render prompt V1 or V2 around one utterance (verbatim, no trimming)."""
    block = render_definitions_block()
    if version == "V1":
        return (
            "You will route one customer request to the correct handler.\n"
            "\n"
            "The 16 handlers are:\n"
            f"{block}\n"
            "\n"
            "Request:\n"
            '"""\n'
            f"{utterance}\n"
            '"""\n'
            "\n"
            "Reply with exactly one handler name from the list above — nothing else."
        )
    if version == "V2":
        return (
            "Which of the following handlers should take this request?\n"
            "\n"
            "Handlers and their criteria:\n"
            f"{block}\n"
            "\n"
            "The request:\n"
            '"""\n'
            f"{utterance}\n"
            '"""\n'
            "\n"
            "Answer with a single handler name and nothing else."
        )
    raise ValueError(f"unknown prompt version {version!r} (expected one of {PROMPT_VERSIONS})")