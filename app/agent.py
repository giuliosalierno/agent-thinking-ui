# ruff: noqa
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

"""A test agent that streams deterministic 'agent thoughts'.

Instead of an LLM, this uses a custom ``BaseAgent`` that yields a scripted,
fully reproducible sequence of intermediate *thought* events followed by a
final answer. Each thought part carries ``thought=True``, which ADK's A2A layer
serializes as ``adk_thought: true`` — the flag the My Google / Gemini
Enterprise UI uses to render the "Show thinking" button and dropdown.

Try these prompts in ``agents-cli playground`` to see different behaviours:
  - "weather in SF"          -> orchestration fallback + tool call + JSON (GOOD)
  - "show my role profile"   -> the doc's canonical GOOD example, verbatim
  - "list holidays"          -> multi-step tool flow with JSON in-block
  - "bad block example"      -> reproduces the doc's BAD example (label -> "{")
  - "long thought / truncate"-> over-long status line (UI truncation)
"""

import asyncio
import os
from collections.abc import AsyncGenerator

from google.adk.agents import BaseAgent

from google.adk.agents.invocation_context import InvocationContext
from google.adk.apps import App
from google.adk.events import Event

from app.thoughts import final_event, steps_for, thought_event

# Artificial delay between streamed steps so a human can read each thought block
# as it appears in the UI. Tune per deploy without a code change via
# `agents-cli deploy --update-env-vars THOUGHT_STEP_DELAY_SECONDS=2`; set to 0
# to stream with no pause.
STEP_DELAY_SECONDS = float(os.environ.get("THOUGHT_STEP_DELAY_SECONDS", "1.5"))


def _last_user_text(ctx: InvocationContext) -> str:
    """Best-effort extraction of the user's message text."""
    content = getattr(ctx, "user_content", None)
    if content and content.parts:
        return " ".join(p.text for p in content.parts if p.text)
    return ""


class ThoughtStreamingAgent(BaseAgent):
    """Streams scripted thought blocks, then a final answer.

    Deterministic by design: the same prompt always yields the same sequence,
    so the documented UI parsing rules (block extraction, good/bad examples,
    truncation) can be observed and asserted reproducibly.
    """

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        query = _last_user_text(ctx)
        for i, (kind, text) in enumerate(steps_for(query)):
            # Pause before each step (except the first) so the UI reveals the
            # thought blocks one at a time at a human-readable pace.
            if i > 0 and STEP_DELAY_SECONDS > 0:
                await asyncio.sleep(STEP_DELAY_SECONDS)
            if kind == "thought":
                yield thought_event(self.name, text, ctx.invocation_id)
            else:
                yield final_event(self.name, text, ctx.invocation_id)


root_agent = ThoughtStreamingAgent(name="root_agent")

app = App(
    root_agent=root_agent,
    name="app",
)
