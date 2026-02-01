from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

from langgraph.graph import StateGraph, START, END, MessagesState

from langgraph.prebuilt import tools_condition, ToolNode



def multiply(a:int,b:int) -> int:
    """Multiplies a and b.

    Args:
        a: first int
        b: second int
    """
    print("multiply")
    return a*b

def add(a:int, b:int):
    """adds a and b.

    Args:
        a: first int
        b: second int
    """
    print("aadding")
    return a+b
def toolCalling(state:MessagesState):
    return {"messages":[llm_with_tool.invoke(state["messages"])]}

model_id = "meta-llama/Llama-3.3-70B-Instruct"

llm = HuggingFaceEndpoint(repo_id=model_id, temperature=0.7)
llm = ChatHuggingFace(llm=llm)
llm_with_tool = llm.bind_tools([multiply,add])

builder = StateGraph(MessagesState)
builder.add_node("toolcalling",toolCalling)
builder.add_node("tools",ToolNode([multiply,add]))
builder.add_edge(START,"toolcalling")
builder.add_conditional_edges("toolcalling",tools_condition)
builder.add_edge("tools",END)

graph = builder.compile()
answer = graph.invoke({"messages":["what is 3 add 2 and then multiply 4?"]})

print(answer)