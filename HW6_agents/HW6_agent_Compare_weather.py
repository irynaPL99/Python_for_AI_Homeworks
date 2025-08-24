"""agent:
Compare the weather forecast for the next three days (starting from today, {current_date}) "
    "for Hamburg, Rostock, and Neustadt in Holstein, Germany.
"""
# Import standard module for handling date and time
from datetime import datetime, timedelta
# Import Tool class for creating custom tools
from langchain_core.tools import Tool
# Import Google Generative AI model for language processing
from langchain_google_genai import ChatGoogleGenerativeAI
# Import Tavily search tool for retrieving web data
from langchain_community.tools.tavily_search import TavilySearchResults
# Import HumanMessage for creating user messages
from langchain_core.messages import HumanMessage
# Import MemorySaver for preserving conversation state
from langgraph.checkpoint.memory import MemorySaver
# Import function to create a ReAct agent
from langgraph.prebuilt import create_react_agent
# Import dotenv for loading environment variables
from dotenv import load_dotenv
# Import os for managing environment variables
import os
# Import Path for handling file paths
from pathlib import Path
# Import requests for making HTTP API calls
import requests

# Load environment variables from .env file located one folder up
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

# Verify and set API keys for Google, Tavily, and OpenWeatherMap
google_api_key = os.getenv("GEMINI_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")
openweather_api_key = os.getenv("OPENWEATHER_API_KEY")
if not google_api_key or not tavily_api_key or not openweather_api_key:
    raise ValueError("GEMINI_API_KEY, TAVILY_API_KEY, or OPENWEATHER_API_KEY not found in .env file")

os.environ["GOOGLE_API_KEY"] = google_api_key
os.environ["TAVILY_API_KEY"] = tavily_api_key

# Initialize the Google Generative AI model
try:
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
except Exception as e:
    raise ValueError(f"Failed to initialize LLM: {str(e)}")

# Initialize memory to save conversation state
memory = MemorySaver()

# Initialize Tavily search tool with a limit of 3 results for more comprehensive data
search = TavilySearchResults(max_results=3)

# Function to get the current date in ISO format
def get_current_date(*args, **kwargs):
    return datetime.now().isoformat()

# Create a tool for retrieving the current date
date_tool = Tool(
    name="Datetime",
    func=get_current_date,
    description="Returns current datetime in ISO format."
)

# Function to get weather forecast for a city
def get_weather_forecast(city: str, *args, **kwargs) -> str:
    """Fetches 3-day weather forecast for a given city using OpenWeatherMap API."""
    base_url = "http://api.openweathermap.org/data/2.5/forecast"
    params = {
        "q": f"{city},DE",  # City name with country code (DE for Germany)
        "appid": openweather_api_key,
        "units": "metric",  # Use Celsius for temperature
        "cnt": 24  # Get 3 days of data (8 intervals/day * 3 days = 24)
    }
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()

        # Process forecast data for the next 3 days
        forecast = []
        current_date = datetime.now().date()
        for i in range(3):
            target_date = (current_date + timedelta(days=i)).strftime("%Y-%m-%d")
            daily_forecasts = [entry for entry in data["list"] if entry["dt_txt"].startswith(target_date)]
            if not daily_forecasts:
                continue

            # Get average temperature, precipitation probability, and wind speed for the day
            avg_temp = sum(entry["main"]["temp"] for entry in daily_forecasts) / len(daily_forecasts)
            precip_prob = max((entry.get("pop", 0) * 100) for entry in daily_forecasts)  # Convert to percentage
            wind_speed = sum(entry["wind"]["speed"] for entry in daily_forecasts) / len(daily_forecasts)

            forecast.append(
                f"{target_date}: Avg Temp: {avg_temp:.1f}°C, Precipitation: {precip_prob:.0f}% chance, "
                f"Wind: {wind_speed:.1f} m/s"
            )
        return f"Weather forecast for {city}:\n" + "\n".join(forecast)
    except Exception as e:
        return f"Error fetching weather for {city}: {str(e)}"

# Create a tool for retrieving weather forecasts
weather_tool = Tool(
    name="WeatherForecast",
    func=get_weather_forecast,
    description="Fetches 3-day weather forecast for a given city in Germany, including temperature, precipitation, and wind conditions."
)

# Combine all tools into a list
tools = [search, date_tool, weather_tool]

# Create the ReAct agent with the LLM, tools, and memory
agent_executor = create_react_agent(llm, tools, checkpointer=memory)

# Configuration for the agent with a unique thread ID
config = {"configurable": {"thread_id": "weather_compare_123"}}

# Define the query to compare weather forecasts for the next three days
query = (
    "Compare the weather forecast for the next three days (starting from today, {current_date}) "
    "for Hamburg, Rostock, and Neustadt in Holstein, Germany. "
    "Provide details on temperature, precipitation, and wind conditions for each city. "
    "Use the WeatherForecast tool to fetch the data."
).format(current_date=get_current_date())

# Execute the agent with the weather comparison query
try:
    for step in agent_executor.stream(
        {"messages": [HumanMessage(content=query)]},
        config,
        stream_mode="values",
    ):
        # Print the latest message from each step
        step["messages"][-1].pretty_print()
except Exception as e:
    raise ValueError(f"Error executing agent: {str(e)}")
