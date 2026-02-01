from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langgraph.graph import START, StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain.messages import HumanMessage

from langgraph.graph import MessagesState
model_id = "meta-llama/Llama-3.3-70B-Instruct"

llm = HuggingFaceEndpoint(repo_id=model_id,temperature=0)

llm = ChatHuggingFace(llm=llm)


def mul(a:int, b:int):
    "mul takes 2 argument a and b, returns multiplication of those to numbers."
    return a*b


llm_with_tool = llm.bind_tools([mul])

def toool_call(state:MessagesState):

    return {"messages":[llm_with_tool.invoke(state["messages"])]}

def custom_decision(state:MessagesState):
    lastm = state["messages"][-1]
    print(lastm)
    if "tool_calls" in lastm.additional_kwargs:
        print("toolCall")
        return "tools"
    else:
        return END
builder = StateGraph(MessagesState)
builder.add_node("toolcalling", toool_call)
builder.add_node("tools",ToolNode([mul]))
builder.add_edge(START,"toolcalling")
builder.add_conditional_edges("toolcalling",tools_condition)# custom_decision can be call instead of tools_condition.
builder.add_edge("tools",END)

graph = builder.compile()

messages = [HumanMessage(content="3 multiply by 2.")]
messages = graph.invoke({"messages":messages})

for m in messages["messages"]:
    m.pretty_print()


messages = [HumanMessage(content="Hello")]
messages = graph.invoke({"messages":messages})

for m in messages["messages"]:
    m.pretty_print()