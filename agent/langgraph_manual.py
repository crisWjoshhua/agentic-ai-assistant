import os

from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage

from langgraph.graph import MessagesState, StateGraph, START
from langgraph.prebuilt import ToolNode, tools_condition
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver

from tools.calculate import calculate
from tools.weather import get_weather
from tools.search_policy import search_policy


# ==================================================
# ENVIRONMENT
# ==================================================

load_dotenv()
MEMORY_DB_PATH = os.getenv(
    "MEMORY_DB_PATH",
    "conversation_memory.db"
)

# Using the second API key because the first key
# reached its Gemini quota during our testing.
api_key = os.getenv("GEMINI_API_KEY6")

if not api_key:
    raise ValueError("GEMINI_API_KEY is not configured")


# ==================================================
# MODEL
# ==================================================

model = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    google_api_key=api_key
)


# ==================================================
# STATE
# ==================================================

class AgentState(MessagesState):
    pass


# ==================================================
# TOOLS
# ==================================================

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


# ==================================================
# MODEL WITH TOOLS
# ==================================================

model_with_tools = model.bind_tools(tools)

tool_node = ToolNode(tools)


# ==================================================
# MODEL NODE
# ==================================================

def call_model(state: AgentState):

    system_message = SystemMessage(
        content="""
You are a helpful AI assistant.

When answering company policy questions, use the information
returned by the policy_search tool.

Do not invent company policy information.

Give the user a clear, concise answer based on the retrieved
policy information.

Do not include internal tool calls, distances, chunk IDs,
metadata, or technical details in the answer.

Use the previous conversation messages to understand
follow-up questions and maintain conversation context.
"""
    )

    response = model_with_tools.invoke(
        [system_message] + state["messages"]
    )

    return {
        "messages": [response]
    }


# ==================================================
# BUILD GRAPH
# ==================================================

builder = StateGraph(AgentState)

builder.add_node("model", call_model)

builder.add_node("tools", tool_node)


# START → MODEL

builder.add_edge(
    START,
    "model"
)


# MODEL → TOOLS or END

builder.add_conditional_edges(
    "model",
    tools_condition
)


# TOOLS → MODEL

builder.add_edge(
    "tools",
    "model"
)


# ==================================================
# PERSISTENT MEMORY
# ==================================================

connection = sqlite3.connect(
    MEMORY_DB_PATH,
    check_same_thread=False
)

memory = SqliteSaver(connection)

agent = builder.compile(
    checkpointer=memory
)

# ==================================================
# TEXT EXTRACTION
# ==================================================

def extract_text(content):

    # Normal string response
    if isinstance(content, str):
        return content

    # Gemini may return a list of content blocks
    if isinstance(content, list):

        text_parts = []

        for block in content:

            if isinstance(block, dict):

                if block.get("type") == "text":

                    text_parts.append(
                        block.get("text", "")
                    )

        return "\n".join(text_parts)

    return str(content)


# ==================================================
# INTERACTIVE CONVERSATION
# ==================================================

if __name__ == "__main__":

    # --------------------------------------------------
    # Conversation identity
    # --------------------------------------------------

    config = {
        "configurable": {
            "thread_id": "conversation_1"
        }
    }


    print("\n======================================")
    print("       AI ASSISTANT STARTED")
    print("======================================")

    print("\nType 'exit' or 'quit' to stop.\n")


    # --------------------------------------------------
    # Conversation loop
    # --------------------------------------------------

    while True:

        user_input = input("You: ")


        # --------------------------------------------------
        # Exit
        # --------------------------------------------------

        if user_input.lower() in ["exit", "quit"]:

            print("\nGoodbye!")

            break


        # --------------------------------------------------
        # Send message to LangGraph
        # --------------------------------------------------

        result = agent.invoke(
            {
                "messages": [
                    ("user", user_input)
                ]
            },
            config=config
        )


        # --------------------------------------------------
        # Get latest assistant response
        # --------------------------------------------------

        answer = extract_text(
            result["messages"][-1].content
        )


        print("\nAssistant:")
        print(answer)

        print()