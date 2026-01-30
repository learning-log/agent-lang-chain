from langgraph.graph import MessagesState
from langgraph.graph import START, END, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from typing_extensions import TypedDict

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

class State(TypedDict):
    graph_state:str

def multiply(state) -> int:
    """Multiplies a and b.

    Args:
        a: first int
        b: second int
    """
    print("multiply")
    return state

def add(state):
    """adds a and b.

    Args:
        a: first int
        b: second int
    """
    print("aadding")
    return state
def decision_maker(state:State):
    if state["graph_state"] == "add":
        return "add"
    else:
        print("multiply")
        return "multiply"
    



model_id = "meta-llama/Llama-3.3-70B-Instruct"

# llm = HuggingFaceEndpoint(repo_id=model_id, temperature=0.7)
# llm = ChatHuggingFace(llm=llm)

# llm_with_tools = llm.bind_tools([multiply,add])

def toolcalling(state:MessagesState):
    print("toolcalling")
    # state["graph_state"] = "add"
    return state

builder = StateGraph(State)
builder.add_node("tool_calling",toolcalling)
builder.add_node("add",add)
builder.add_node("multiply",multiply)
builder.add_edge(START,"tool_calling")
builder.add_conditional_edges("tool_calling",decision_maker)
builder.add_edge("add",END)
builder.add_edge("multiply",END)
graph = builder.compile()
graph.invoke({"graph_state":"multiply"})

