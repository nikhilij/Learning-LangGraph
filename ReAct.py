import os
from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langchain_openai import AzureChatOpenAI
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

load_dotenv()

# Annotated - provides metadata about the type

email = Annotated[str, "this is has to be a valid email format"]

print(email.__metadata__)

# Sequence -> to automatically handle the state updates for sequences such as by adding new messages to chat history

# dotenv -> load env variables from the process

# BaseMessage-> the foundational class for all message types in LangGraph

# ToolMessage -> passes data back to LLM after it call a tool such as the content and the tool_call_id

# SystemMessage for providing instructions to the LLM

# add_message -> is a reducer function , rule that control how updates from nodes are combined with the exisiting state, it tells us how to merge new data into the current state,
# without a reducer, updates would have replced the existing value entirely


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


@tool
def add(a: int, b: int):
    """This is an addition function that adds two numbers"""
    return a + b


tools = [add]

model = AzureChatOpenAI(
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
).bind_tools(tools)


def model_call(state: AgentState) -> AgentState:
    System_prompt = SystemMessage(
        content="You are my Ai assistant, please answer my query to the best of your ability."
    )
    response = model.invoke([System_prompt] + state["messages"])
    return {"messages": [response]}


def should_continue(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]
    if not last_message.tool_calls:
        return "End"
    else:
        return "continue"


graph = StateGraph(AgentState)
graph.add_node("Our_agent", model_call)

tool_node = ToolNode(tools=tools)
graph.add_node("Tool_Node", tool_node)

graph.set_entry_point("Our_agent")
graph.add_conditional_edges(
    "Our_agent",
    should_continue,
    {
        "continue": "Tool_Node",
        "End": END,
    },
)

graph.add_edge("Tool_Node", "Our_agent")

app = graph.compile()

png_data = app.get_graph().draw_mermaid_png()
with open("graph2.png", "wb") as f:
    f.write(png_data)
print("Graph saved as 'graph2.png'. Open it to view the visualization.")
print("\nMermaid code (copy to a Mermaid viewer):")



def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()


input = {"messages": [HumanMessage(content="Add 5 and 7")]}
print_stream(app.stream(input, stream_mode="values"))
