# Our Imports
import asyncio
import functools
import time
import uuid
from dotenv import load_dotenv

from google.adk.sessions import DatabaseSessionService
from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.agents.run_config import RunConfig
from google.adk.agents.invocation_context import LlmCallsLimitExceededError
from google.adk.workflow._errors import DynamicNodeFailError
from google.adk.models.lite_llm import LiteLlm
from google.genai import types
from google.adk.events.event import Event
from google.adk.events.event_actions import EventActions

# We are going to create our own recipe assistant!

load_dotenv()

DB_URL = "sqlite+aiosqlite:///./recipe_data.db"  # persistent storage file
APP_NAME = "Personal Recipe Assistant"
USER_ID = "alice"


def _answer_from_tool(fn):
    """Stop after one tool call. qwen3:8b otherwise calls the same tool again."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        result = fn(*args, **kwargs)
        tool_context = kwargs.get("tool_context")
        if tool_context is not None and hasattr(tool_context, "actions"):
            tool_context.actions.skip_summarization = True
        return result

    return wrapper


def _remember_recipes(tool_context, recipes):
    """Write the recipe list and keep the count in session state."""
    recipes = list(recipes)
    tool_context.state["recipes"] = recipes
    tool_context.state["total_recipes"] = len(recipes)
    return recipes


def _load_recipes(tool_context):
    if tool_context and hasattr(tool_context, "state"):
        return list(tool_context.state.get("recipes", []))
    return []


def _collection_state(state):
    state = dict(state or {})
    recipes = list(state.get("recipes") or [])
    return {
        "chef_name": state.get("chef_name") or "Alice",
        "recipes": recipes,
        "total_recipes": len(recipes),
    }


def _session_has_leaked_reasoning(session) -> bool:
    for event in getattr(session, "events", None) or []:
        if getattr(event, "error_code", None) or getattr(event, "error_message", None):
            return True
        content = getattr(event, "content", None)
        for part in getattr(content, "parts", None) or []:
            text = getattr(part, "text", None) or ""
            stripped = text.strip()
            if stripped in ("{}", "[]") or stripped.startswith("<think>"):
                return True
            if stripped.startswith("{") or stripped.startswith("["):
                return True
            if "Okay, the user" in text:
                return True
    return False


# All of the tools our agent will have access to


# Adding a recipe
@_answer_from_tool
def add_recipe(
    name: str,
    ingredients: str,
    instructions: str,
    cook_time: str = "30 minutes",
    tool_context=None,
) -> dict:
    """Add a new recipe to your collection"""
    recipe = {
        "name": name,
        "ingredients": ingredients,
        "instructions": instructions,
        "cook_time": cook_time,
        "date_added": time.strftime("%Y-%m-%d"),
    }

    if tool_context and hasattr(tool_context, "state"):
        # Use tool_context.state if available (proper ADK way)
        recipes = list(tool_context.state.get("recipes", []))
        recipes.append(recipe)
        _remember_recipes(tool_context, recipes)
        print(f"[Recipes] Added recipe: {name}")
        print_recipe_summary(recipes)
        return {
            "action": "add_recipe",
            "recipe": recipe,
            "total_recipes": len(recipes),
            "message": f"Added recipe: {name}",
        }
    else:
        # Fallback: manually update session state and persist to database
        session = service.get_session(
            app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID
        )
        recipes = session.state.get("recipes", [])
        recipes.append(recipe)

        # Create an event with state delta to persist changes
        from google.adk.sessions import types as session_types

        event = Event(
            id=str(uuid.uuid4()),
            author="user",
            content=types.Content(
                role="user", parts=[types.Part(text=f"Add recipe: {name}")]
            ),
            actions=session_types.Actions(state_delta={"recipes": recipes}),
            timestamp=time.time(),
            turn_complete=True,
        )

        # Append the event to persist state changes
        service.append_event(session=session, event=event)

        print(f"[Recipes] Added recipe: {name}")
        print_recipe_summary(recipes)
        return {
            "action": "add_recipe",
            "recipe": name,
            "message": f"Added recipe: {name}",
        }


# Viewing all recipes
@_answer_from_tool
def view_recipes(tool_context=None) -> dict:
    """View all saved recipes"""
    recipes = _load_recipes(tool_context)
    print_all_recipes(recipes)
    if not recipes:
        message = "No recipes saved yet"
    else:
        message = f"{len(recipes)} recipe(s) saved"
    return {
        "action": "view_recipes",
        "total_recipes": len(recipes),
        "recipes": recipes,
        "message": message,
    }


# Getting a specific recipe
@_answer_from_tool
def get_recipe(name: str, tool_context=None) -> dict:
    """Get details of a specific recipe by name"""
    recipes = _load_recipes(tool_context)

    # Find recipe (case insensitive)
    for recipe in recipes:
        if recipe["name"].lower() == name.lower():
            print_recipe_details(recipe)
            return {
                "action": "get_recipe",
                "recipe": recipe,
                "message": f"Found recipe: {name}",
            }

    print(f"[Recipes] Recipe '{name}' not found")
    return {
        "action": "get_recipe",
        "recipe": name,
        "message": f"Recipe '{name}' not found",
    }


# Deleting a recipe
@_answer_from_tool
def delete_recipe(name: str, tool_context=None) -> dict:
    """Delete a recipe by name"""
    if tool_context and hasattr(tool_context, "state"):
        recipes = list(tool_context.state.get("recipes", []))

        # Find and remove recipe
        for i, recipe in enumerate(recipes):
            if recipe["name"].lower() == name.lower():
                removed = recipes.pop(i)
                _remember_recipes(tool_context, recipes)

                print(f"[Recipes] Deleted recipe: {name}")
                print_recipe_summary(recipes)
                return {
                    "action": "delete_recipe",
                    "recipe": removed,
                    "total_recipes": len(recipes),
                    "message": f"Deleted recipe: {name}",
                }

        print(f"[Recipes] Recipe '{name}' not found")
        return {
            "action": "delete_recipe",
            "recipe": name,
            "message": f"Recipe '{name}' not found",
        }
    else:
        # Fallback: manually update session state and persist to database
        session = service.get_session(
            app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID
        )
        recipes = session.state.get("recipes", [])

        # Find and remove recipe
        for i, recipe in enumerate(recipes):
            if recipe["name"].lower() == name.lower():
                removed = recipes.pop(i)

                # Create an event with state delta to persist changes
                from google.adk.sessions import types as session_types

                event = Event(
                    id=str(uuid.uuid4()),
                    author="user",
                    content=types.Content(
                        role="user", parts=[types.Part(text=f"Delete recipe: {name}")]
                    ),
                    actions=session_types.Actions(state_delta={"recipes": recipes}),
                    timestamp=time.time(),
                    turn_complete=True,
                )

                # Append the event to persist state changes
                service.append_event(session=session, event=event)

                print(f"[Recipes] Deleted recipe: {name}")
                print_recipe_summary(recipes)
                return {
                    "action": "delete_recipe",
                    "recipe": name,
                    "message": f"Deleted recipe: {name}",
                }

        print(f"[Recipes] Recipe '{name}' not found")
        return {
            "action": "delete_recipe",
            "recipe": name,
            "message": f"Recipe '{name}' not found",
        }


# Search recipes by ingredient
@_answer_from_tool
def search_recipes(ingredient: str, tool_context=None) -> dict:
    """Search for recipes containing a specific ingredient"""
    recipes = _load_recipes(tool_context)

    # Find recipes containing the ingredient
    matching_recipes = []
    for recipe in recipes:
        if ingredient.lower() in recipe["ingredients"].lower():
            matching_recipes.append(recipe)

    if matching_recipes:
        print(f"[Search] Found {len(matching_recipes)} recipe(s) with '{ingredient}':")
        for recipe in matching_recipes:
            print(f"  • {recipe['name']} ({recipe['cook_time']})")
    else:
        print(f"[Search] No recipes found with ingredient '{ingredient}'")

    return {
        "action": "search_recipes",
        "ingredient": ingredient,
        "found": len(matching_recipes),
        "recipes": matching_recipes,
        "message": (
            f"Found {len(matching_recipes)} recipe(s) with '{ingredient}'"
            if matching_recipes
            else f"No recipes found with ingredient '{ingredient}'"
        ),
    }


# Defining our agent here!
recipe_agent = Agent(
    name="recipe_agent",
    # reasoning_effort="none" turns off Qwen3 thinking so the reply is the
    # answer, not the scratchpad.
    model=LiteLlm(model="ollama/qwen3:8b", reasoning_effort="none"),
    description="Personal recipe collection assistant",
    instruction="""
You manage a personal recipe collection. Call at most one tool, then answer in one or two plain sentences. Never output JSON or your reasoning.

Tools:
- add_recipe(name, ingredients, instructions, cook_time) — only when the user gave ingredients and instructions. Otherwise ask for them.
- view_recipes() — list saved recipes
- get_recipe(name) — one recipe
- delete_recipe(name)
- search_recipes(ingredient)

Chef: {chef_name}
Saved recipes: {total_recipes}
""",
    tools=[add_recipe, view_recipes, get_recipe, delete_recipe, search_recipes],
)

# Creating our session state
service = DatabaseSessionService(db_url=DB_URL)
initial_state = {"chef_name": "Alice", "total_recipes": 0, "recipes": []}

# We'll initialize the session inside an async function
SESSION_ID = None
runner = None


async def initialize_session():
    """Initialize session and runner - must be called from async context"""
    global SESSION_ID, runner

    # Seeing if we have a pre-existing session (ASYNC!)
    resp = await service.list_sessions(app_name=APP_NAME, user_id=USER_ID)
    if resp.sessions:
        existing = await service.get_session(
            app_name=APP_NAME,
            user_id=USER_ID,
            session_id=resp.sessions[0].id,
        )
        state = _collection_state(existing.state if existing else None)
        # Earlier turns stored Qwen's scratchpad as the assistant reply.
        # Keep the recipes and start a clean conversation.
        if existing and _session_has_leaked_reasoning(existing):
            await service.delete_session(APP_NAME, USER_ID, existing.id)
            session = await service.create_session(
                app_name=APP_NAME,
                user_id=USER_ID,
                state=state,
            )
            SESSION_ID = session.id
            print(
                f"Started a fresh session with {state['total_recipes']} saved recipe(s):",
                SESSION_ID,
            )
        else:
            SESSION_ID = resp.sessions[0].id
            if existing and existing.state.get("total_recipes") != state["total_recipes"]:
                await service.append_event(
                    session=existing,
                    event=Event(
                        invocation_id=str(uuid.uuid4()),
                        author="recipe_agent",
                        actions=EventActions(
                            state_delta={"total_recipes": state["total_recipes"]}
                        ),
                    ),
                )
            print("Continuing session:", SESSION_ID)
    else:
        # Create session (ALSO ASYNC!)
        session = await service.create_session(
            app_name=APP_NAME,
            user_id=USER_ID,
            state=initial_state,
        )
        SESSION_ID = session.id
        print("Created new session:", SESSION_ID)

    # Creating our runner
    runner = Runner(agent=recipe_agent, app_name=APP_NAME, session_service=service)


# Helper functions for printing recipes - FOR DEBUGGING PURPOSES
def print_recipe_summary(recipes):
    if not recipes:
        print("[Recipes] No recipes saved yet")
    else:
        print(f"[Recipes] Total: {len(recipes)} recipe(s)")
        for recipe in recipes:
            print(
                f"  • {recipe['name']} ({recipe['cook_time']}) - added {recipe['date_added']}"
            )


def print_all_recipes(recipes):
    if not recipes:
        print("[Recipes] No recipes saved yet")
    else:
        print(f"\n[Recipe Collection] {len(recipes)} recipe(s):")
        for i, recipe in enumerate(recipes, 1):
            print(f"\n{i}. {recipe['name']}")
            print(f"   Cook time: {recipe['cook_time']}")
            print(f"   Added: {recipe['date_added']}")


def print_recipe_details(recipe):
    print(f"\n[Recipe Details] {recipe['name']}")
    print(f"Cook time: {recipe['cook_time']}")
    print(f"Added: {recipe['date_added']}")
    print(f"\nIngredients:\n{recipe['ingredients']}")
    print(f"\nInstructions:\n{recipe['instructions']}")


def _reply_text(part) -> str | None:
    """User-facing text. Skip thought traces and JSON blobs."""
    if getattr(part, "thought", False):
        return None
    text = getattr(part, "text", None)
    if not isinstance(text, str):
        return None
    if "</think>" in text:
        text = text.split("</think>", 1)[-1]
    text = text.strip()
    if not text or text.startswith("<think>") or text[0] in "{[":
        return None
    return text


def _speak_tool_result(result: dict) -> str:
    """Plain reply when the model echoes JSON or calls tools in a loop."""
    action = result.get("action")
    recipes = result.get("recipes") or []
    if action == "view_recipes":
        if not recipes:
            return "You don't have any saved recipes yet."
        names = ", ".join(recipe["name"] for recipe in recipes)
        return f"You have {len(recipes)} saved recipe(s): {names}."
    if action == "add_recipe":
        name = result.get("recipe", {}).get("name", "that recipe")
        return f"Added {name}. You now have {result.get('total_recipes', 0)} saved recipe(s)."
    if action == "get_recipe":
        recipe = result.get("recipe")
        if isinstance(recipe, dict):
            return (
                f"{recipe['name']} takes {recipe.get('cook_time', 'some time')}. "
                f"Ingredients: {recipe.get('ingredients', '')}. "
                f"Instructions: {recipe.get('instructions', '')}"
            )
        return result.get("message") or "I couldn't find that recipe."
    if action == "delete_recipe":
        return result.get("message") or "Done."
    if action == "search_recipes":
        if not recipes:
            return result.get("message") or "No matching recipes."
        names = ", ".join(recipe["name"] for recipe in recipes)
        return f"Recipes with {result.get('ingredient')}: {names}."
    return result.get("message") or "Done."


# Function to call the agent
async def ask_agent(text: str):
    msg = types.Content(role="user", parts=[types.Part(text=text)])
    printed = False
    last_tool = None

    try:
        async for ev in runner.run_async(
            user_id=USER_ID,
            session_id=SESSION_ID,
            new_message=msg,
            # qwen3:8b will call the same tool again unless this is capped.
            run_config=RunConfig(max_llm_calls=3),
        ):
            for response in ev.get_function_responses() or []:
                payload = getattr(response, "response", None)
                if isinstance(payload, dict) and payload.get("action"):
                    last_tool = payload
            if ev.is_final_response() and ev.content and ev.content.parts:
                for part in ev.content.parts:
                    reply = _reply_text(part)
                    if reply:
                        print("\n👩‍🍳 Recipe Assistant:", reply)
                        printed = True
    except (LlmCallsLimitExceededError, DynamicNodeFailError):
        pass

    if not printed and last_tool:
        print("\n👩‍🍳 Recipe Assistant:", _speak_tool_result(last_tool))


# Our loop to chat with our agent
async def main():
    # Initialize session first!
    await initialize_session()

    print("\n🍳 Personal Recipe Assistant ready! (type 'quit' to exit)")
    print(
        "Try: 'add pasta recipe', 'view recipes', 'search chicken', 'get pasta recipe'\n"
    )

    while True:
        q = input("You: ")
        if q.lower() in ("quit", "exit"):
            print("Recipe collection saved. Happy cooking! 👋")
            break
        await ask_agent(q)


if __name__ == "__main__":
    asyncio.run(main())