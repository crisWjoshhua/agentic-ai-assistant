import os

from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import MessagesState
from langgraph.prebuilt import ToolNode, tools_condition

from tools.calculate import calculate
from tools.weather import get_weather
from tools.search_policy import search_policy


# --------------------------------------------------
# ENVIRONMENT
# --------------------------------------------------

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY2")

if not api_key:
    raise ValueError("GEMINI_API_KEY2 is not configured")


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


tools = [
    calculator,
    weather,
    policy_search
]


# --------------------------------------------------
# MODEL
# --------------------------------------------------

model = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    google_api_key=api_key
)

model_with_tools = model.bind_tools(tools)


# --------------------------------------------------
# MODEL NODE
# --------------------------------------------------

def call_model(state: MessagesState):

    response = model_with_tools.invoke(state["messages"])

    return {
        "messages": [response]
    }


# --------------------------------------------------
# TOOL NODE
# --------------------------------------------------

tool_node = ToolNode(tools)


# --------------------------------------------------
# GRAPH
# --------------------------------------------------

builder = StateGraph(MessagesState)


# Add nodes
builder.add_node("model", call_model)
builder.add_node("tools", tool_node)


# Starting point
builder.add_edge(START, "model")


# Decide whether to use a tool or finish
builder.add_conditional_edges(
    "model",
    tools_condition
)


# After tools execute, go back to model
builder.add_edge("tools", "model")


# Compile graph
agent = builder.compile()


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

    query = (
        "Calculate 25 * 20 and tell me the current weather "
        "in Chennai and tell me if the company policy says "
        "if I can take my children to the office."
    )

    answer = run_agent(query)

    print("\nAI RESPONSE:")
    print(answer)