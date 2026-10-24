import datetime
from zoneinfo import ZoneInfo
from google.adk.agents import Agent, SequentialAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools import google_search

# AGENT_MODEL = "ollama/qwen3:8b"
AGENT_MODEL = "openai/gpt-5-nano"


def get_weather(city: str) -> dict:
    """ Retrieves the current weather for a given city

    Args: 
        dict: A dictionary containing the weather information for the given city
    """
    print(f"--- Tool: get_weather called for city: {city} ---")
    city_normalized = city.lower().replace(" ", "") # Basic normalization

    mock_weather_db = {
        "newyork": {
            "status": "success",
            "report": "The weather in New York is sunny and 60 degrees Fahrenheit."
        },
        "losangeles": {
            "status": "success",
            "report": "The weather in Los Angeles is sunny and 70 degrees Fahrenheit."
        },
        "chicago": {
            "status": "success",
            "report": "The weather in Chicago is cloudy and 50 degrees Fahrenheit."
        },
    }
    
    if city_normalized not in mock_weather_db:
        return {
            "status": "error",
            "error": f"No weather data found for city: {city}"
        }
    
    return mock_weather_db[city_normalized]

def get_current_time(city: str) -> dict:
    """ Returns the current time in a specified city.

    Args:
        city (str): The name of the city for which to retrieve the curren time.

    Returns:
        dict: status and result or error msg.
    """

    if city.lower() == "new york":
        tz_identifier = "America/New_York"
    else:
        return {
            "status": "error",
            "error_message": (f"Sorry, I dont have timezone information for {city}")
        }
    
    tz = ZoneInfo(tz_identifier)
    now = datetime.datetime.now(tz)
    report = f'The current time in {city} is {now.strftime("%Y-%m-%d %H:%M:%S %Z%z")}'
    return {"status":"sucess","report":report}

destination_research_agent = Agent(
    name="DestinationResearchAgent",
    model=AGENT_MODEL,
    tools=[google_search],
    description="An agent that researches travel destinations and gathers essential information",
    instruction="""
    You are a travel researcher. You will be given a destination and travel preferences, and you will research:
    - Best time to visit and weather patterns
    - Top attractions and must-see locations
    - Local culture, customs, and etiquette tips
    - Transportation options within the destination
    - Safety considerations and travel requirements
    Provide comprehensive destination insights for trip planning.
    """,
    output_key="destination_research",
)

itinerary_builder_agent = Agent(
    name="ItineraryBuilderAgent",
    model=AGENT_MODEL,
     description="An agent that creates structured travel itineraries with daily schedules",
    instruction="""
    You are a professional travel planner. Using the research from "destination_research" output, create a detailed itinerary that includes:
    - Day-by-day schedule with recommended activities
    - Suggested accommodation areas or districts
    - Estimated time requirements for each activity
    - Meal recommendations and dining suggestions
    - Budget estimates for major expenses
    Structure it logically for easy following during the trip.
    """,
    output_key="travel_itinerary",
)

travel_optimizer_agent = Agent(
    model=AGENT_MODEL,
    name="TravelOptimizerAgent",
    description="An agent that optimizes travel plans with practical advice and alternatives",
    instruction="""
    You are a seasoned travel consultant. Using the itinerary from "travel_itinerary" output, optimize it by adding:
    - Money-saving tips and budget alternatives
    - Packing recommendations specific to the destination
    - Backup plans for weather or unexpected situations
    - Local apps, websites, or resources to download
    - Cultural do's and don'ts for respectful travel
    
    Format the final output as:
    
    ITINERARY: {travel_itinerary}
    
    OPTIMIZATION TIPS: [your money-saving and practical tips here]
    
    TRAVEL ESSENTIALS: [packing and preparation advice here]
    
    BACKUP PLANS: [alternative options and contingencies here]
    """,
)

# root_agent = Agent(
#     name="travel_planner_agent",
#     model=LiteLlm(model=AGENT_MODEL),
#     description="An agent that helps users plan plan their travel itinerary",
#     instruction="You are a travel planner agent. Help the user to plan an itinerary for traveling",
#     tools=[get_weather, get_current_time]
# )



root_agent = SequentialAgent(
    name="TravelPlanningSystem",
    # model=LiteLlm(AGENT_MODEL),#not needed for SequentialAgent
    # model=AGENT_MODEL, #not needed for SequentialAgent
    description="A comprehensive system that researches destinations, builds itineraries, and optimizes travel plans",
    sub_agents=[
        destination_research_agent,
        itinerary_builder_agent,
        travel_optimizer_agent,
    ],
    # instruction="You are a travel planner agent. Help the user plan their trip.",
    # tools=[get_weather, get_current_time],
)