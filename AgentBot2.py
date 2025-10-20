from typing import TypedDict, List,Union
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import AzureChatOpenAI
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
from IPython.display import display, Image

import os

load_dotenv()


class AgentState(TypedDict):
    messages: List[Union[HumanMessage, AIMessage]]


llm = AzureChatOpenAI(
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
)


def process(state: AgentState) -> AgentState:
    response = llm.invoke(state["messages"])
    state["messages"].append(AIMessage(content=response.content))
    print(f"\nAI : {response.content}")
    # print("current state:", state["messages"])

    return state

graph = StateGraph(AgentState)
graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END)
agent = graph.compile()

conversation_history = []

user_input = input("Enter: ")
while user_input != "exit":
    try:
        conversation_history.append(HumanMessage(content=user_input))
        result = agent.invoke({"messages": conversation_history})
        conversation_history = result["messages"]
        user_input = input("Enter: ")
    except EOFError:
        print("\nExiting due to end of input.")
        break


with open("logging.txt",'w') as file:
    file.write("your conversation Log:\n")
    for message in conversation_history:
        if isinstance(message,HumanMessage):
            file.write(f"You: {message.content}\n")
        elif isinstance(message,AIMessage):
            file.write(f"AI: {message.content}\n")

    file.write("\nEnd of conversation.\n")

print("Conversation log saved to 'logging.txt'.")

















#     print(f"\nAI : {response.content}")
#     return state


# graph = StateGraph(AgentState)
# graph.add_node("process", process)
# graph.add_edge(START, "process")
# graph.add_edge("process", END)
# agent = graph.compile()

# # Display or save the graph
# png_data = agent.get_graph().draw_mermaid_png()
# with open("graph.png", "wb") as f:
#     f.write(png_data)
# print("Graph saved as 'graph.png'. Open it to view the visualization.")
# print("\nMermaid code (copy to a Mermaid viewer):")

# user_input = input("Enter: ")

# while user_input != "exit":
#     try:
#         agent.invoke({"messages": [HumanMessage(content=user_input)]})
#         user_input = input("Enter: ")
#     except EOFError:
#         print("\nExiting due to end of input.")
#         break
