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

"""Local verification of the 'agent thoughts' streaming behaviour.

Runs the agent through a real ADK ``Runner`` for a few prompts and, for every
emitted part, shows:
  - whether it is a thought (genai ``Part.thought``),
  - the ACTUAL A2A serialization (proving ``adk_thought: true`` is emitted),
  - the collapsed "Show thinking" button label the UI would display at that
    point (via the documented block-extraction algorithm).

Run:  uv run python scripts/verify_thoughts.py
"""

from __future__ import annotations

import warnings

from google.adk.a2a.converters.part_converter import convert_genai_part_to_a2a_part
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from app.agent import root_agent
from app.thoughts import build_thought_stream, extract_thinking_label

warnings.filterwarnings("ignore")  # silence ADK A2A "experimental" notices

PROMPTS = [
    "weather in SF",
    "show my role profile",
    "list holidays",
    "bad block example",
    "long thought please truncate",
]


def _thought_flag(part: types.Part) -> bool:
    return bool(getattr(part, "thought", None))


def run_prompt(prompt: str) -> None:
    print("\n" + "=" * 72)
    print(f"PROMPT: {prompt!r}")
    print("=" * 72)

    session_service = InMemorySessionService()
    session = session_service.create_session_sync(user_id="u", app_name="verify")
    runner = Runner(
        agent=root_agent, session_service=session_service, app_name="verify"
    )
    message = types.Content(role="user", parts=[types.Part.from_text(text=prompt)])

    emitted_blocks: list[str] = []
    for event in runner.run(
        new_message=message, user_id="u", session_id=session.id
    ):
        if not (event.content and event.content.parts):
            continue
        for part in event.content.parts:
            if part.text is None:
                continue
            is_thought = _thought_flag(part)
            a2a = convert_genai_part_to_a2a_part(part)
            meta = getattr(a2a.root, "metadata", None) or {}
            first_line = part.text.splitlines()[0] if part.text.splitlines() else ""

            if is_thought:
                emitted_blocks.append(part.text)
                label = extract_thinking_label(build_thought_stream(emitted_blocks))
                print(f"\n  [THOUGHT] first line: {first_line!r}")
                print(f"            a2a metadata: {meta}")
                print(f"            -> UI button label now: {label!r}")
            else:
                print(f"\n  [ANSWER ] {part.text!r}")
                print(f"            a2a metadata: {meta}  (no adk_thought)")


def main() -> None:
    for prompt in PROMPTS:
        run_prompt(prompt)
    print("\n" + "=" * 72)
    print("Done. Thought parts carry {'adk_thought': True}; answers do not.")
    print("=" * 72)


if __name__ == "__main__":
    main()
