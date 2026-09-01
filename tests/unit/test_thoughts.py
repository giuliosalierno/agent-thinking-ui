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
    SEARCH_CALL_ID,
    build_thought_stream,
    extract_thinking_label,
    final_event,
    function_call_event,
    function_response_event,
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


def test_function_call_and_response_serialize_as_tool_parts() -> None:
    """A call/response pair must serialize as the A2A tool DataParts the UI
    turns into a chip (+ "✓"), keyed on the built-in ``google_search`` name."""
    call = function_call_event(
        "root_agent", "google_search", {"query": "x"}, call_id="c1"
    )
    call_a2a = convert_genai_part_to_a2a_part(call.content.parts[0])
    assert call_a2a.root.metadata.get("adk_type") == "function_call"
    assert call_a2a.root.data["name"] == "google_search"

    resp = function_response_event(
        "root_agent", "google_search", {"ok": True}, call_id="c1"
    )
    resp_a2a = convert_genai_part_to_a2a_part(resp.content.parts[0])
    assert resp_a2a.root.metadata.get("adk_type") == "function_response"
    # Shared id is what correlates the response with its call.
    assert call.content.parts[0].function_call.id == "c1"
    assert resp.content.parts[0].function_response.id == "c1"


def test_search_scenario_shape() -> None:
    """The search scenario emits: preamble, a labelled thought, a matched
    google_search call/response pair, then the final answer."""
    steps = steps_for("give info to the Gemini Enterprise roadmap")
    kinds = [k for k, _ in steps]
    assert kinds == ["final", "thought", "call", "response", "final"]

    # The thought's first line is what the UI shows as the collapsed label.
    label = next(p for k, p in steps if k == "thought")
    assert extract_thinking_label(label) == "Search Gemini Roadmap"

    call = next(p for k, p in steps if k == "call")
    resp = next(p for k, p in steps if k == "response")
    assert call["name"] == resp["name"] == "google_search"
    assert call["id"] == resp["id"] == SEARCH_CALL_ID


def test_scenario_routing() -> None:
    assert select_scenario("give info to the Gemini Enterprise roadmap") == "search"
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
