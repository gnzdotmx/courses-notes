# Module 2: ADK Masterclass - Deep Dive

## Objectives

- ADK is Google’s framework for building, running, and deploying agents the way you build software
- **Model-agnostic** and **deployment-agnostic**; the smooth path is Gemini and Google
- An agent reasons with a model, or a workflow agent controls execution
- Session holds one conversation; memory recalls a user across conversations
- Local loop: CLI and dev UI to run, inspect, and evaluate, then deploy

## Key concepts

| Term | Definition / notes |
| ---- | ------------------ |
| ADK (Agent Development Kit) | Google framework to **build, manage, evaluate, and deploy** conversational and non-conversational agents |
| Agent | Worker for **one task**. Uses a language model to reason, or controls other agents |
| Workflow agent | **Deterministic** controller of execution: sequential, parallel, loop |
| Tool | Lets an agent **act**: call an API, search, run code, hit a database, or call another agent |
| Callback | Your code at a **fixed point** in the agent process: checks, logging, behavior changes |
| Session | One conversation: **history, events, and working memory** |
| State | Working memory **inside that session** |
| Memory | Facts about a user **across sessions** |
| Session service | Owns **short-term** state for a session |
| Orchestration | The kit’s wiring so agents, tools, and services **run as one unit** |
| Evaluation | Measure how an agent behaves, including **multi-turn** datasets, locally |

## What ADK is

ADK is one kit for the pieces that are usually split across frameworks. Agent work is meant to feel like ordinary software: create, orchestrate, deploy. Scope runs from a single task to a multi-agent workflow.

| Property | What it means |
| -------- | ------------- |
| Model-agnostic | Plug in the model you have. Gemini is the favored path |
| Deployment-agnostic | Run on the server you choose. Google services are the simple deploy path |
| Compatible | Fits next to other agent frameworks you already use |

Under the pillars: **debugging, tracing, and the model** that drives the agent.

## Two kinds of agent

An agent either **reasons** (language model) or **directs** (workflow agent).

Sequential, parallel, and loop agents are workflow agents. Used when the path should stay fixed. They solve predictable pipelines. LLM-driven dynamic routing sits beside them when the next agent depends on the situation. Together you get fixed pipelines and adaptive behavior.

| Agent | What it is | Name cue |
| ----- | ---------- | -------- |
| Sequential | Sub-agents run in a fixed order | One **after** another |
| Parallel | Sub-agents run together | At the **same time** |
| Loop | A step runs again | **Repeats** |
| LLM-driven routing | The model picks the next agent or transfer | Path is **decided at runtime** |

Delegation is either an **LLM-driven transfer** or an **explicit agent call**. A called agent can still use a tool (API, service, database) to fetch context.

## Pillars

| Piece | What it is for |
| ----- | -------------- |
| Agent | The worker for a specific task |
| Tool | Action beyond the reply: APIs, search, code, other services, other agents |
| Callback | Snippet that runs at a chosen moment |
| Session + state | Context of the current conversation |
| Memory | User context from earlier conversations |
| Evaluation | Performance and what happened inside the run |
| Deployment | Put the agent on a platform so it can run at scale |
| Orchestration | Makes the pieces talk; most of that stays behind the SDK |

Orchestration is the part you do not wire by hand. You call the SDK and the kit runs the moving parts.

## Session, state, and memory

Session and state are one conversation. Memory is every conversation with that user.

| | Session + state | Memory |
| - | --------------- | ------ |
| Scope | **One** conversation | **Many** sessions, one user |
| Holds | History, events, working memory (state) | Facts worth recalling later |
| Owner | Session service | A long-term memory service you plug in |
| Use | The agent can continue this chat | A chatbot can pick up who the user is next time |

## Key capabilities

| Capability | What you get |
| ---------- | ------------ |
| Multi-agent design | Specialized agents in a **hierarchy**. They split a task and delegate |
| Tool ecosystem | Custom functions, **agents as tools**, code execution, search, databases, APIs |
| Flexible orchestration | Built-in workflow agents **and** LLM routing |
| Developer tooling | **CLI** and a **dev UI**: run locally, inspect steps, events, state changes, debug visually |
| LLM support | Gemini first; other models plug in |
| State and memory | Short-term state in the session service; long-term memory as an integration point |
| Evaluation | Build multi-turn datasets; run them locally in the CLI or the dev UI |

## Build loop

Define the goal before the agents. Interactions follow from that goal.

```text
goal and scope
  → design agents and how they interact
    → attach tools
      → connect agents to the rest of the system
        → run, debug, evaluate locally
          → wrong: back to design
          → right: deploy
```

## Tools to install

Any IDE works. The course demos use VS Code. Python is required. Dependency installs in the course use **uv** (pip is fine if you already use it).

Have **one** model key before the labs. Google AI Studio covers basic Gemini models on the free tier. OpenAI and Anthropic keys are there when an exercise uses those models.

| Tool | Role |
| ---- | ---- |
| VS Code | Editor used in the demos |
| Python | Language the agents are written in |
| uv | Dependency manager the course uses |
| Google AI Studio key | Gemini, including free basic models |
| OpenAI key | Optional; some exercises |
| Anthropic key | Optional; some exercises |
