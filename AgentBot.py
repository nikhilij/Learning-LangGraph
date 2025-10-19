from typing import TypedDict, List
from langchain_core.messages import HumanMessage
from langchain_openai import AzureChatOpenAI
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
from IPython.display import display, Image

import os

load_dotenv()


class AgentState(TypedDict):
    messages: List[HumanMessage]


llm = AzureChatOpenAI(
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
)


def process(state: AgentState) -> AgentState:
    response = llm.invoke(state["messages"])
    print(f"\nAI : {response.content}")
    return state


graph = StateGraph(AgentState)
graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END)
agent = graph.compile()

# Display or save the graph
png_data = agent.get_graph().draw_mermaid_png()
with open("graph.png", "wb") as f:
    f.write(png_data)
print("Graph saved as 'graph.png'. Open it to view the visualization.")
print("\nMermaid code (copy to a Mermaid viewer):")

user_input = input("Enter: ")

while user_input != "exit":
    try:
        agent.invoke({"messages": [HumanMessage(content=user_input)]})
        user_input = input("Enter: ")
    except EOFError:
        print("\nExiting due to end of input.")
        break
