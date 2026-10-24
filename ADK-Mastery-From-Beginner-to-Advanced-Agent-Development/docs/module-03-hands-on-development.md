# Module 3: Hands-on Development

## Objectives

- A package layout ADK can import, so one file becomes a runnable agent
- Which model string and key go where, so the same agent can call Ollama, OpenAI, or Gemini
- How a function becomes a tool the model is allowed to call
- Where LLM agents, workflow agents, and custom agents sit in the class tree
- Which command gives the dev UI, the HTTP API, or a terminal chat

## Key concepts

| Term | Definition / notes |
| ---- | ------------------ |
| `root_agent` | The **instance ADK loads** from the agent package |
| `Agent` | Alias of `LlmAgent`: **name, model, description, instruction, tools** |
| `LiteLlm` | Wraps a **provider/model string** so LiteLLM can call Ollama or OpenAI |
| Tool | A **function object** on `tools`. The model calls it; your code does not |
| Tool docstring | Text the model reads to decide **when to call** and **what shape comes back** |
| `.env` | API keys in the **agent package folder**. Never commit or share them |
| `BaseAgent` | **Parent class** of every agent type |
| Workflow agent | **Fixed control flow**: sequential, parallel, or loop |
| `adk web` | Local **dev UI** on a FastAPI server |
| `adk run` | **Terminal chat** against one agent package |
| Event | One step of a run: user text, model config, **function call**, or reply |
| Session | One conversation, keyed by a **session id** in the dev UI |

## Project layout

ADK loads a Python package that defines `root_agent`. `uv` owns the environment.

```text
adk_agents/                 # uv init
  pyproject.toml
  multi_agents/
    __init__.py             # from . import agent
    agent.py                # root_agent = Agent(...)
    .env                    # keys for this package only
```

| Piece | Rule |
| ----- | ---- |
| `multi_agents/` | Package name ADK is given. Convention from the ADK docs |
| `__init__.py` | `from . import agent` so importing the package loads `agent.py` |
| `agent.py` | Must assign `root_agent` |
| `.env` | Lives **inside** `multi_agents/`, next to `agent.py` |

```bash
uv init
uv venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
uv add google-adk
uv add litellm                     # required even though LiteLlm is imported from google.adk
```

`google-adk` does not install LiteLLM. `from google.adk.models.lite_llm import LiteLlm` fails at runtime until `litellm` is a project dependency.

## The agent

```python
from google.adk.agents import Agent

root_agent = Agent(
    name="travel_planner_agent",   # id in the UI; use underscores
    model="gemini-2.0-flash",
    description="Helps the user plan a trip.",
    instruction="You are a travel planner. Help the user plan their trip.",
    tools=[],
)
```

| Argument | What it does |
| -------- | ------------ |
| `name` | Agent id. Underscores, not spaces |
| `model` | Who writes the reply. String for Gemini; `LiteLlm(...)` for other providers |
| `description` | Short label of what this agent is |
| `instruction` | System instruction. Shows up in the event request |
| `tools` | Functions the model may call. Empty until you attach one |

`Agent(...)` and `LlmAgent(...)` are the same class. Use either.

## Model providers

Ollama through LiteLLM, OpenAI through LiteLLM, and a Gemini model string are backends for the same `Agent`. Used when you want a local model, another vendor's API, or Gemini. They solve which model writes the reply.

| Backend | What it is | Why it is needed | When it is used | How you notice |
| ------- | ---------- | ---------------- | --------------- | -------------- |
| Ollama + `LiteLlm` | `LiteLlm(model="ollama/gemma3")` | Run **without an API key** | The weights are on your machine | Events list the Ollama model id |
| OpenAI + `LiteLlm` | `LiteLlm(model="openai/gpt-5-nano")` | Call an **OpenAI** model | You have `OPENAI_API_KEY` | Events list `openai/...` |
| Gemini | `model="gemini-2.0-flash"` | **Native** ADK path. No wrapper | You have `GOOGLE_API_KEY` | Events list Gemini. Trace shows the invocation |

Name cue: **LiteLLM** is the thin adapter in front of many providers. **Gemini** is passed as the model id itself, so the wrapper stays off.

```python
from google.adk.models.lite_llm import LiteLlm

model = LiteLlm(model="ollama/gemma3")      # local
model = LiteLlm(model="openai/gpt-5-nano")  # OpenAI
# Gemini: drop LiteLlm and pass the string
model = "gemini-2.0-flash"
```

| Provider | Setup |
| -------- | ----- |
| Ollama | Install Ollama, `ollama pull gemma3`, `ollama list` to confirm, daemon running |
| OpenAI | `OPENAI_API_KEY` in `multi_agents/.env` |
| Gemini | `GOOGLE_API_KEY` from [Google AI Studio](https://aistudio.google.com/) (Get API key). `GOOGLE_GENAI_USE_VERTEXAI=FALSE` so ADK uses that key, not Vertex AI |

```bash
# multi_agents/.env — values stay out of git
OPENAI_API_KEY=...
GOOGLE_GENAI_USE_VERTEXAI=FALSE
GOOGLE_API_KEY=...
```

Changing `model` is the whole switch. Name, instruction, and tools stay put. Restart `adk web` after the change.

## Run the agent

`adk web`, `adk api_server`, and `adk run` start the same agent. Used when you want a UI, an HTTP server, or a terminal. They solve talking to the agent while you build it.

Run from the directory that **contains** `multi_agents/`.

| Command | What it is | Why it is needed | When it is used | How you notice |
| ------- | ---------- | ---------------- | --------------- | -------------- |
| `adk web` | FastAPI **plus** the dev UI | Inspect events, state, and the graph | First run, and any time you debug a tool call | Browser URL, session id, panels |
| `adk api_server` | FastAPI **without** the UI | Same backend, no panels | You only want the HTTP API | Server process, no dev UI |
| `adk run multi_agents` | Interactive **CLI** | Chat without a browser | A quick check of a reply or a tool | `user:` prompt. Type `exit` to quit |
| `adk create` | Scaffolds a new app | Starter files in the current folder | A new app, not this package | New bare-bones project |
| `adk deploy` | Sends the agent to a **host** | Run it off your machine | A hosted environment | Remote endpoint |
| `adk eval` | Runs **evaluation** cases | Measure behavior | You already have eval data | Eval output |

Name cue: **web** is the UI, **api_server** is the server alone, **run** is the terminal session.

```bash
adk --help
adk web
adk run multi_agents
```

## What the dev UI shows

| Panel | What you read |
| ----- | ------------- |
| Session id | Which conversation this is. State hangs off this id |
| Events | User text, system instruction, model id, function call (`id`, name, args), function response, token counts |
| Trace | Invocation timing. In this walkthrough it filled in once the model was Gemini |
| State | Working memory for this session. Empty until something writes it |
| Artifacts | Files the run saved. Empty until something produces one |
| Graph | This agent and every tool on `tools`. One agent is one node; the node grows as you register tools |

Use Events to confirm the model id and whether a tool actually ran. The reply text alone does not show that.

## Tools

A tool is a function you **register**, not a call you make. The docstring is the model's spec: what it does, the arguments, and the dict it returns. Mock data is enough to see the call. A real tool would put an API request where the dict lookup is.

```python
import datetime
from zoneinfo import ZoneInfo

from google.adk.agents import Agent


def get_weather(city: str) -> dict:
    """Current weather for a city.

    Args:
        city (str): City name, for example Tokyo.

    Returns:
        dict: status "success" and a report, or status "error" and error_message.
    """
    # Toy table. Not a weather service.
    reports = {"tokyo": "Light rain, 18 C", "london": "Cloudy, 15 C"}
    key = city.lower().replace(" ", "")
    if key not in reports:
        return {"status": "error", "error_message": f"No data for {city}."}
    return {"status": "success", "report": reports[key]}


def get_current_time(city: str) -> dict:
    """Current local time for a city.

    Args:
        city (str): City name. This toy only knows New York.

    Returns:
        dict: status and report, or status and error_message.
    """
    if city.lower() != "new york":
        return {"status": "error", "error_message": f"No timezone for {city}."}
    tz = ZoneInfo("America/New_York")  # IANA id, not a display name
    now = datetime.datetime.now(tz)
    report = f"The current time in {city} is {now.strftime('%Y-%m-%d %H:%M:%S %Z%z')}"
    return {"status": "success", "report": report}


root_agent = Agent(
    name="travel_planner_agent",
    model="gemini-2.0-flash",
    description="Helps the user plan a trip.",
    instruction="You are a travel planner. Help the user plan their trip.",
    tools=[get_weather, get_current_time],  # objects, not get_weather()
)
```

| Return key | When |
| ---------- | ---- |
| `status` | `"success"` or `"error"` on every return |
| `report` | The fact the model should say, on success |
| `error_message` | Why it failed, on error |

Call flow:

- User text arrives with the system instruction and the tool docstrings.
- The model emits a function call: name, id, arguments (the city).
- ADK runs that function and feeds the dict back.
- The model writes the user-facing sentence from `report` or `error_message`.
- Events lists the call and the response. The graph lists every name in `tools`.

The model picks the tool. "Weather in Tokyo" calls `get_weather`. "What time is it in New York?" calls `get_current_time`. A trip question with no weather or time need gets a normal reply and no tool call.

A function that is not in `tools` does not exist for the agent. Define it, then add it to the list, then restart.

`get_weather` and `get_current_time` above are toys: a handful of cities, and a clock only for `America/New_York`. The lab copy lives in `code/adk_agents/multi_agents/agent.py`.

## Agent types

`LlmAgent`, workflow agents, and a custom `BaseAgent` subclass are the primary agent types. Used when the next step is a model decision, a fixed pipeline, or logic you write yourself. They solve who runs and in what order.

All of them extend `BaseAgent`.

| Type | What it is | Why it is needed | When it is used | How you notice |
| ---- | ---------- | ---------------- | --------------- | -------------- |
| `LlmAgent` (`Agent`) | One model, plus instruction and tools | The agent has to **reason** | A single worker, like this travel planner | One node. Replies and tool calls come from the model |
| `SequentialAgent` | Sub-agents run in a **fixed order**, each handing off to the next | The path must not depend on the model | A pipeline of specialists | Output of one is the input of the next |
| `ParallelAgent` | Sub-agents run **at the same time**, then join | Independent work should not wait in line | Fan-out, then one combined result | Several branches finish, then one result |
| `LoopAgent` | A step **repeats** until a stop condition | Keep going until the result is good enough | Retry or refine | The same step appears again, then the loop exits |
| Custom agent | Your subclass of `BaseAgent` | Built-in types cannot express the control flow | Rare; most apps stay on LLM plus workflow agents | You wrote the run logic |

Name cue: **sequential** is one after another, **parallel** is at the same time, **loop** is the same step again.

This module only builds the `LlmAgent`. Sequential and parallel agents are the next module.
