# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Deterministic 'agent thoughts' streaming helpers.

This module reproduces the *My Google / Gemini Enterprise* "Show thinking"
behaviour described in go/unified-ai:agent-thoughts, but for the Python ADK
stack (the design doc's ``ToolThoughtGenerator`` / ``BoqletModule`` examples are
Java/Boqlet).

Key fact, verified against the installed ADK: a ``genai`` ``Part`` with
``thought=True`` is serialized by ADK's A2A layer with the metadata flag
``adk_thought: true`` (``adk_`` prefix + ``thought`` key, see
``google.adk.a2a.converters.part_converter``). That is exactly the
``ADK_A2A_THOUGHT_KEY`` the UI keys on.

Modelling rule used throughout:
    ONE thought ``Event`` == ONE block.
    - Put verbose context (e.g. JSON) on *subsequent lines of the same event's
      text* to keep the action line as the collapsed button label (GOOD).
    - Emit that verbose context as its *own* event and it becomes a separate
      block whose first line (e.g. ``{``) becomes the label (BAD).
"""

from __future__ import annotations

import json
from collections.abc import Callable

from google.adk.events import Event
from google.genai import types

# Mirrors the Java ToolThoughtGenerator CALL_TEMPLATES / RESPONSE_TEMPLATES maps:
# tool name -> present-participle status line (action-oriented, ends with "...").
CALL_TEMPLATES: dict[str, str] = {
    "list_holidays": "Getting holiday dates...",
    "get_weather": "Checking the weather...",
    "get_role_profile": "Getting your role profile details...",
}
RESPONSE_TEMPLATES: dict[str, str] = {
    "list_holidays": "Fetched holiday dates.",
    "get_weather": "Got the latest conditions.",
    "get_role_profile": "Fetched role profile.",
}

# Generic orchestration/routing fallback (doc best practice).
ORCHESTRATION_FALLBACK = "Thinking..."

# The UI's default when no label can be parsed from the stream.
DEFAULT_LABEL = "Show thinking"

# Blocks in the accumulated thought stream are separated by a double newline.
BLOCK_SEPARATOR = "\n\n"


def extract_thinking_label(stream: str, *, fallback: str = DEFAULT_LABEL) -> str:
    """Reproduce the UI's collapsed-button label extraction (local simulation).

    Per the design doc, the UI: splits the accumulated stream into blocks
    separated by double newlines, takes the *latest* block, and extracts the
    first non-empty line of that block. If nothing can be parsed, it falls back
    to ``"Show thinking"``.

    This is a faithful reimplementation of the *documented* algorithm so the
    good/bad/truncation examples can be asserted locally without the real UI.
    """
    if not stream or not stream.strip():
        return fallback
    blocks = stream.split(BLOCK_SEPARATOR)
    latest = blocks[-1]
    for line in latest.splitlines():
        if line.strip():
            return line.strip()
    return fallback


def build_thought_stream(blocks: list[str]) -> str:
    """Join emitted thought blocks the way the UI accumulates them."""
    return BLOCK_SEPARATOR.join(blocks)


def thought_event(author: str, text: str, invocation_id: str = "") -> Event:
    """Build an intermediate *thought* Event (one block).

    The ``thought=True`` flag is what ADK's A2A converter turns into the
    ``adk_thought: true`` metadata the UI recognizes.

    ``invocation_id`` must be the current invocation's id: the managed
    (Vertex AI) session service rejects appended events without it
    (``event.invocation_id; Required field is not set``). The in-memory
    session service used by the playground does not enforce this.
    """
    return Event(
        invocation_id=invocation_id,
        author=author,
        content=types.Content(
            role="model",
            parts=[types.Part(text=text, thought=True)],
        ),
    )


def final_event(author: str, text: str, invocation_id: str = "") -> Event:
    """Build the final, user-facing answer Event (NOT a thought)."""
    return Event(
        invocation_id=invocation_id,
        author=author,
        content=types.Content(
            role="model",
            parts=[types.Part(text=text)],
        ),
    )


# --- Scenarios -------------------------------------------------------------
# A Step is ("thought" | "final", text). Each "thought" step is one block.

Step = tuple[str, str]


def _weather_lookup(query: str) -> dict[str, str]:
    if "sf" in query.lower() or "san francisco" in query.lower():
        return {"city": "San Francisco", "temp": "60F", "condition": "foggy"}
    return {"city": query.strip() or "your area", "temp": "90F", "condition": "sunny"}


def weather_scenario(query: str) -> list[Step]:
    result = _weather_lookup(query)
    return [
        # Generic orchestration fallback while routing.
        ("thought", ORCHESTRATION_FALLBACK),
        # Tool-call thought (present participle + ellipsis).
        ("thought", CALL_TEMPLATES["get_weather"]),
        # Tool-response thought: verbose JSON on subsequent lines of the SAME
        # block, so the label stays "Analyzing weather data..." (GOOD example).
        ("thought", "Analyzing weather data...\n" + json.dumps(result, indent=2)),
        (
            "final",
            f"It's currently {result['condition']} and {result['temp']} "
            f"in {result['city']}.",
        ),
    ]


def profile_scenario(_query: str) -> list[Step]:
    # Reproduces the doc's canonical GOOD example verbatim.
    result = {"role_title": "Software Engineer", "cost_center": "Engineering"}
    return [
        ("thought", CALL_TEMPLATES["get_role_profile"]),
        ("thought", "Analyzing role profile...\n" + json.dumps(result, indent=2)),
        (
            "final",
            "Your role is Software Engineer in the Engineering cost center.",
        ),
    ]


def holiday_scenario(_query: str) -> list[Step]:
    result = [
        "2026-01-01 New Year's Day",
        "2026-07-04 Independence Day",
        "2026-12-25 Christmas Day",
    ]
    return [
        ("thought", ORCHESTRATION_FALLBACK),
        ("thought", CALL_TEMPLATES["list_holidays"]),
        ("thought", "Compiling holiday list...\n" + json.dumps(result, indent=2)),
        (
            "final",
            "The upcoming holidays are New Year's Day, Independence Day, "
            "and Christmas Day.",
        ),
    ]


def bad_block_scenario(_query: str) -> list[Step]:
    # Demonstrates the doc's BAD example: emitting the JSON as its OWN block
    # (separate thought event) makes the UI label read "{".
    result = {"role_title": "Software Engineer", "cost_center": "Engineering"}
    return [
        ("thought", "Getting your role profile details..."),
        ("thought", "Analyzing role profile..."),
        ("thought", json.dumps(result, indent=2)),  # own block -> label "{"
        (
            "final",
            "Your role is Software Engineer in the Engineering cost center.",
        ),
    ]


def long_line_scenario(_query: str) -> list[Step]:
    # Over-long status line to demonstrate UI truncation with ellipses.
    return [
        ("thought", ORCHESTRATION_FALLBACK),
        (
            "thought",
            "Searching the entire corporate knowledge base across every region "
            "and business unit for anything even tangentially related to your "
            "unusually specific and very long question...",
        ),
        ("final", "Here is what I found."),
    ]


SCENARIOS: dict[str, Callable[[str], list[Step]]] = {
    "weather": weather_scenario,
    "profile": profile_scenario,
    "holiday": holiday_scenario,
    "bad": bad_block_scenario,
    "long": long_line_scenario,
}


def select_scenario(query: str) -> str:
    """Pick a scenario from the user's message (keyword routing)."""
    q = query.lower()
    if "profile" in q or "role" in q:
        return "profile"
    if "holiday" in q:
        return "holiday"
    if "bad" in q:
        return "bad"
    if "long" in q or "truncat" in q:
        return "long"
    return "weather"


def steps_for(query: str) -> list[Step]:
    return SCENARIOS[select_scenario(query)](query)
