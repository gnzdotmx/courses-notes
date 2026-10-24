# Module 7: Wrap Up & Next Steps

## Objectives

- The stack these notes cover: agent, tools, workflow, session, memory, callbacks
- The layer ADK runs on each request: tool calls, sessions, memory, orchestration
- The moves: write session state before the agent, drop a finished session, test locally, then deploy
- External model keys, the `model` field that switches provider, and what Events and `after_agent_callback` expose
- Which surface can move while the kit is still changing, and which split stays
- Where to build the next agent, and where the current docs and changelog live

## Key concepts

| Term | Definition / notes |
| ---- | ------------------ |
| ADK | Google’s **Agent Development Kit**. One kit for the agent, its tools, and the runtime around a request |
| Core split | An agent **reasons or routes**, a tool **acts**, and ADK **runs the request**. That split stays when APIs move |
| Orchestration | **Coordinates agents and tools** in one workflow. You call the SDK; the kit orders the run |
| `before_agent_callback` | **Sets or updates session state** before the agent runs. Incoming-request metadata goes here |
| `after_agent_callback` | Runs after the agent. The context has **metadata, state, input, and output** |
| API key | Credential for an **external model or API**: OpenAI, Anthropic, Google AI Studio |
| Model switch | Change **`model` in the agent config or constructor**. That selects the provider |
| Events | The **sequence and results** of agent operations. In a multi-agent run, how they executed and interacted |
| Long-term memory | Recall **across sessions**. Later turns stay coherent and personal |
| `SequentialAgent` | Sub-agents in a **predefined order**. Each output is the next agent's input |
| `delete_session` | **Removes that session and its state**. Pass `app_name`, `user_id`, and `session_id` |
| `adk deploy` | **Provisions a hosted environment** and deploys the agent there |
| Local test | Run and debug **on your machine**. Agent-logic errors show up here, while the path is still local |
| Moving surface | Imports, flags, and UI labels **can change** while ADK is still in development. Recheck the repo before copying an old snippet |
| Practice loop | Skill comes from **building agents in ADK**, then reading the docs for the piece that failed |

## What you can reach for

| Piece | Use it for | Notes |
| ----- | ---------- | ----- |
| Agent + model | One task that reasons | [Module 3](module-03-hands-on-development.md) |
| Tool | An action the model asks for: API, search, code, another agent | [Module 3](module-03-hands-on-development.md) |
| Workflow agent | A fixed order, or branches that run together | [Module 4](module-04-sequential-and-parallel-agents.md) |
| Session + state | This conversation’s history and working memory | [Module 5](module-05-session-memory-and-structured-outputs.md) |
| Memory | Facts about a user across conversations | [Module 5](module-05-session-memory-and-structured-outputs.md) |
| Callback | A check, a filter, or a log at a fixed point in the request | [Module 6](module-06-callbacks-filtering-and-guardrails.md) |
| Dev UI | Events, trace, state, and the model id for one run | [Module 3](module-03-hands-on-development.md) |
| Evaluation | Multi-turn datasets, run locally, to see how the agent behaves | [Module 2](module-02-adk-masterclass-deep-dive.md) |

## What ADK runs for you

Tool calls, callbacks, session management, memory, and orchestration are the runtime under the agent. Used on any request that needs an action, context, or a controlled step. They solve the glue around the model call.

| | What it is | Why it is needed | When it is used | How you notice |
| --- | ---------- | ---------------- | --------------- | -------------- |
| Tool call | The model names a registered function. ADK runs it and feeds the dict back | The model only emits the call. The kit performs the action | The reply needs data or a side effect | Events shows a function call, then a function response |
| Callback | Your function at one point: agent, model, or tool | Filter, time, or replace that step from code you attach | A turn must be blocked, logged, or rewritten | A refusal with an empty model trace, or a value on state |
| Session | One conversation: history, events, state | The next turn has to see this one | Every `adk web` or runner chat | The same session id. State fills when something writes it |
| Memory | Facts kept for a user after the session ends | A later conversation should still know the user | A memory service is plugged in | A new session still has the stored facts |
| Orchestration | The kit ordering agents, tools, and services for one run | Workflow and multi-agent code stays a declaration | Sequential, parallel, or a model that delegates | The graph and the event order match the declaration |

Name cues: a **callback** is ADK calling your code back at that step. A **session** is one sitting. **Memory** is what remains after that sitting. **Orchestration** is the ordering of the pieces in the run.

## Rules

| When | What you do |
| ---- | ----------- |
| The turn should leave metadata on this session | In `before_agent_callback`, write the incoming request onto `callback_context.state`. That stage can set or update session data. Return `None` and the agent still runs |
| The user should be known on a later conversation | Plug in a long-term memory service. The agent recalls earlier sessions, so later turns stay coherent and personal |
| The answer needs live data, such as weather in a city | Register a tool. It fetches **accurate, current** data from an external source |
| Steps must run in order and hand work forward | Use a `SequentialAgent`. Sub-agents run in the predefined order and pass each output to the next |
| Several agents and tools share one workflow | Leave the coordination to orchestration. It manages that workflow |
| You are still building | Test locally. Errors in the agent logic show up before a database, a host, or another service is on the path |
| The session service is a database | Call `load_dotenv()` before the connection. Those variables are the credentials and the connection settings |
| The session is finished | `delete_session(app_name=..., user_id=..., session_id=...)`. That call removes the session and its state. See [Module 5](module-05-session-memory-and-structured-outputs.md) |
| The agent should leave your machine | `adk deploy` provisions a hosted environment and deploys the agent there |
| The agent calls an external model or API | Set the key first: OpenAI, Anthropic, or Google AI Studio. Those credentials are how the workflow reaches that model or API. See [Module 3](module-03-hands-on-development.md) |
| You want a different provider, such as OpenAI to Gemini | Change `model` in the agent config or the constructor. Name, instruction, and tools stay. Restart `adk web` |
| A multi-agent run needs a check | Read Events. The list is the sequence of operations and each result |
| A user deletes a recipe in a database-backed session | The item leaves **session state and persistent storage**. The database updates with the state |
| Work happens after the agent finishes | `after_agent_callback` receives metadata, state, and both the input and the output |

```python
def before_agent_callback(callback_context: CallbackContext):
    # metadata for this request, written before the agent runs
    callback_context.state["request_time"] = datetime.now().isoformat()
    return None  # the agent still runs


def after_agent_callback(callback_context: CallbackContext):
    # metadata, state, and this turn's input and output are on the context
    print(callback_context.state["request_time"])
    return None
```

## What you inspect

| Surface | What you read it for |
| ------- | -------------------- |
| Events | User text, model id, tool name, args, tool result. In a multi-agent run, the **sequence and each result** |
| Trace | How long each step took |
| State | Working memory for this session |
| Model | Which model actually answered. A provider swap shows up here |

Read Events for the tool name, the args, and the model id. The reply text is only the last message.

## What can change

ADK is still under development. Imports, flags, and UI labels can move. The split stays: an agent reasons or routes, a tool acts, a callback guards a step, a session holds this chat, memory holds the user.

| Check | When |
| ----- | ---- |
| [google/adk-python](https://github.com/google/adk-python) | Before you trust an old install, flag, or import |
| The version in your `pyproject.toml` | When a lab snippet and the docs disagree |

## Where to go next

Build an agent that needs one row from the tables above. That is the review.

| Source | What it is for |
| ------ | -------------- |
| [ADK docs](https://google.github.io/adk-docs/) | These topics, plus the advanced pieces past this course |
| [google/adk-python](https://github.com/google/adk-python) | Releases, issues, and what changed in the Python kit |

Use the docs for the version pinned in the project.
