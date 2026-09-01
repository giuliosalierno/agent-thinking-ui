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

"""Deterministic tests for the agent-thoughts streaming behaviour.

These assert on *code contracts* (block parsing, A2A metadata), not LLM output,
so they are appropriate for pytest.
"""

from google.adk.a2a.converters.part_converter import convert_genai_part_to_a2a_part

from app.thoughts import (
    DEFAULT_LABEL,
    build_thought_stream,
    extract_thinking_label,
    final_event,
    select_scenario,
    steps_for,
    thought_event,
)


def test_thought_part_serializes_adk_thought() -> None:
    """A thought part must carry adk_thought:true over A2A; an answer must not."""
    t = thought_event("root_agent", "Searching...")
    a2a = convert_genai_part_to_a2a_part(t.content.parts[0])
    assert a2a.root.metadata == {"adk_thought": True}

    f = final_event("root_agent", "Done.")
    fa2a = convert_genai_part_to_a2a_part(f.content.parts[0])
    assert not (fa2a.root.metadata or {}).get("adk_thought")


def test_good_example_label_is_action_line() -> None:
    """Doc GOOD example: JSON on subsequent lines of the same block keeps the
    action line as the label."""
    stream = build_thought_stream(
        [
            "Getting your role profile details...",
            'Analyzing role profile...\n{\n  "role_title": "Software Engineer"\n}',
        ]
    )
    assert extract_thinking_label(stream) == "Analyzing role profile..."


def test_bad_example_label_is_brace() -> None:
    """Doc BAD example: JSON in its own block makes the label read '{'."""
    stream = build_thought_stream(
        [
            "Getting your role profile details...",
            "Analyzing role profile...",
            '{\n  "role_title": "Software Engineer"\n}',
        ]
    )
    assert extract_thinking_label(stream) == "{"


def test_empty_stream_falls_back() -> None:
    assert extract_thinking_label("") == DEFAULT_LABEL
    assert extract_thinking_label("   \n\n  ") == DEFAULT_LABEL


def test_latest_block_wins() -> None:
    stream = build_thought_stream(["Thinking...", "Checking the weather..."])
    assert extract_thinking_label(stream) == "Checking the weather..."


def test_scenario_routing() -> None:
    assert select_scenario("show my role profile") == "profile"
    assert select_scenario("list holidays") == "holiday"
    assert select_scenario("bad block example") == "bad"
    assert select_scenario("please truncate this long thought") == "long"
    assert select_scenario("weather in SF") == "weather"


def test_steps_shape() -> None:
    """Every scenario ends with exactly one final answer."""
    for prompt in ("weather", "role", "holiday", "bad", "long"):
        steps = steps_for(prompt)
        assert steps, f"no steps for {prompt}"
        assert steps[-1][0] == "final"
        assert sum(1 for kind, _ in steps if kind == "final") == 1
        assert any(kind == "thought" for kind, _ in steps)
