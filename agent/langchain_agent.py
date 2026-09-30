import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool

from tools.calculate import calculate
from tools.weather import get_weather
from tools.search_policy import search_policy


load_dotenv()

api_key = os.getenv("GEMINI_API_KEY2")

if not api_key:
    raise ValueError("GEMINI_API_KEY is not configured")


# --------------------------------------------------
# TOOLS
# --------------------------------------------------

@tool
def calculator(expression: str) -> str:
    """Calculate a mathematical expression."""

    return calculate(expression)


@tool
def weather(city: str) -> str:
    """Get the current weather for a city."""

    return get_weather(city)


@tool
def policy_search(query: str) -> str:
    """Search the company policy documents for relevant information."""

    return search_policy(query)


# --------------------------------------------------
# MODEL
# --------------------------------------------------

model = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    google_api_key=api_key
    
)


# --------------------------------------------------
# AGENT
# --------------------------------------------------

agent = create_agent(
    model=model,
    tools=[
        calculator,
        weather,
        policy_search
    ],
    system_prompt="""
You are a helpful AI assistant.

You have access to three tools:

1. calculator
   Use this for mathematical calculations.

2. weather
   Use this when the user asks for current weather.

3. policy_search
   Use this when the user asks about company policies
   or information contained in company policy documents.

Always use policy_search for company policy questions.
Do not invent company policy information.

If the policy documents do not contain the answer,
clearly tell the user that the information was not found.

Use the appropriate tool whenever one is available.
"""
)


# --------------------------------------------------
# RUN AGENT
# --------------------------------------------------

def run_agent(query: str) -> str:

    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": query
                }
            ]
        }
    )

    content = result["messages"][-1].content

    # Gemini may return content as a list of blocks
    if isinstance(content, list):

        text_parts = []

        for block in content:

            if isinstance(block, dict) and block.get("type") == "text":
                text_parts.append(block.get("text", ""))

            elif isinstance(block, str):
                text_parts.append(block)

        return "\n".join(text_parts).strip()

    return str(content)

# --------------------------------------------------
# TEST
# --------------------------------------------------

if __name__ == "__main__":

    query = "Calculate 25 * 20 and tell me the current weather in Chennai and tell me if the my company policy says if i can take my children to office"

    answer = run_agent(query)

    print("\nAI RESPONSE:")
    print(answer)