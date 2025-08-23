"""Homework 4
1. Добавить защиту от блокировки API (Rate Limits)
Обработать таймауты
Реализовать retry-механизм с tenacity

Библиотека tenacity в Python позволяет автоматически повторять запрос в случае
ошибки. Это полезно при взаимодействии с ненадёжными ресурсами
"""

import time
from tenacity import retry, stop_after_attempt, wait_exponential
from google import genai
from google.genai import types
from dotenv import load_dotenv
import os
from requests import ReadTimeout



# Go one folder up and load .env
from pathlib import Path
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)
#load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# Create Gemini client with timeout (milliseconds)
timeout_seconds = 10
client = genai.Client(
    api_key=api_key,
    http_options=types.HttpOptions(timeout=timeout_seconds * 1000)
)

# Retry if error
# stop_after_attempt(3): try 3 times
# wait_exponential: wait 2 → 4 → 8 sec, max 10 sec
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def get_gemini_response(prompt):
    """
    Send text to Gemini and get answer.
    Handle timeout and errors.
    """

    # Small pause between requests
    time.sleep(0.3)

    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[prompt]
        )
        return response.text
    except ReadTimeout:
        return f"Request timeout ({timeout_seconds} sec)."
    except Exception as e:
        # Handle Rate Limit (429)
        if "429" in str(e):
            return "API limit error. Try again later."
        return f"Error: {str(e)}"


if __name__ == "__main__":
    #response = get_gemini_response("What is request timeout?")
    response = get_gemini_response("Where is Hamburg?")

    if response:
        print(response)
    else:
        print("No answer from Gemini.")
