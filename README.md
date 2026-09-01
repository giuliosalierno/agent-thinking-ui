# agent-thoughts-test

A test agent that streams deterministic **agent thoughts** ("Show thinking")
to reproduce the  Gemini Enterprise UI behaviour on the Python ADK stack.

Agent generated with `agents-cli` version `1.2.0`

## How thoughts work here

A `genai` `Part` with `thought=True` is serialized by ADK's A2A layer with the
metadata flag **`adk_thought: true`** (`adk_` prefix + `thought`, see
`google.adk.a2a.converters.part_converter`) — exactly the `ADK_A2A_THOUGHT_KEY`
the UI keys on. The agent (`app/agent.py`) is a deterministic custom
`BaseAgent` that yields a scripted sequence of thought blocks, then a final
answer. Modelling rule: **one thought event = one block**; verbose JSON on
subsequent lines of the *same* event keeps the action-line label (doc GOOD
example), while JSON in its *own* event makes the label read `{` (doc BAD
example). See `app/thoughts.py`.

Try these prompts in `agents-cli playground`:

| Prompt | Demonstrates |
|--------|--------------|
| `weather in SF` | orchestration fallback + tool call + JSON in-block (GOOD) |
| `show my role profile` | the doc's canonical GOOD example, verbatim |
| `list holidays` | multi-step tool flow with JSON in-block |
| `bad block example` | the doc's BAD example — label collapses to `{` |
| `long thought, truncate` | over-long status line (UI truncation) |

**Verify locally (no GCP needed):**

```bash
uv run pytest tests/unit/test_thoughts.py   # documented parsing rules + adk_thought
uv run python scripts/verify_thoughts.py    # run agent, print adk_thought + UI labels
agents-cli playground                        # click "Show thinking" interactively
```

> Note: the scaffold's `/feedback` endpoint and its `test_collect_feedback`
> e2e test require a real GCP project (Cloud Logging) and will 500 locally —
> unrelated to the agent.

## Project Structure

```
agent-thoughts-test/
├── app/         # Core agent code
│   ├── agent.py               # Main agent logic
│   ├── fast_api_app.py        # FastAPI Backend server
│   └── app_utils/             # App utilities and helpers
├── tests/                     # Unit, integration, and load tests
├── GEMINI.md                  # AI-assisted development guide
└── pyproject.toml             # Project dependencies
```

> 💡 **Tip:** Use [Antigravity CLI](https://antigravity.google/) for AI-assisted development - project context is pre-configured in `GEMINI.md`.

## Requirements

Before you begin, ensure you have:
- **uv**: Python package manager (used for all dependency management in this project) - [Install](https://docs.astral.sh/uv/getting-started/installation/) ([add packages](https://docs.astral.sh/uv/concepts/dependencies/) with `uv add <package>`)
- **agents-cli**: Agents CLI - Install with `uv tool install google-agents-cli`
- **Google Cloud SDK**: For GCP services - [Install](https://cloud.google.com/sdk/docs/install)


## Quick Start

Install `agents-cli` and its skills if not already installed:

```bash
uvx google-agents-cli setup
```

Install required packages:

```bash
agents-cli install
```

Test the agent with a local web server:

```bash
agents-cli playground
```

You can also use features from the [ADK](https://adk.dev/) CLI with `uv run adk`.

## Commands

| Command              | Description                                                                                 |
| -------------------- | ------------------------------------------------------------------------------------------- |
| `agents-cli install` | Install dependencies using uv                                                         |
| `agents-cli playground` | Launch local development environment                                                  |
| `agents-cli lint`    | Run code quality checks                                                               |
| `agents-cli eval`    | Evaluate agent behavior (generate, grade, analyze, and more — see `agents-cli eval --help`) |
| `uv run pytest tests/unit tests/integration` | Run unit and integration tests                                                        || [A2A Inspector](https://github.com/a2aproject/a2a-inspector) | Launch A2A Protocol Inspector                                                        |

## 🛠️ Project Management

| Command | What It Does |
|---------|--------------|
| `agents-cli scaffold enhance` | Add CI/CD pipelines and Terraform infrastructure |
| `agents-cli infra cicd` | One-command setup of entire CI/CD pipeline + infrastructure |
| `agents-cli scaffold upgrade` | Auto-upgrade to latest version while preserving customizations |

---

## Development

Edit your agent logic in `app/agent.py` and test with `agents-cli playground` - it auto-reloads on save.

## Deployment

```bash
gcloud config set project <your-project-id>
agents-cli deploy
```

To add CI/CD and Terraform, run `agents-cli scaffold enhance`.
To set up your production infrastructure, run `agents-cli infra cicd`.

## Observability

Built-in telemetry exports to Cloud Trace, BigQuery, and Cloud Logging.

## A2A Inspector

This agent supports the [A2A Protocol](https://a2a-protocol.org/). Use the [A2A Inspector](https://github.com/a2aproject/a2a-inspector) to test interoperability.
See the [A2A Inspector docs](https://github.com/a2aproject/a2a-inspector) for details.
