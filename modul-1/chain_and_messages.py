from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from typing import TypedDict
from langchain_core.messages import AIMessage, HumanMessage, AnyMessage

from langgraph.graph import START, StateGraph, END

model_id = "meta-llama/Llama-3.3-70B-Instruct"

llm = HuggingFaceEndpoint(repo_id=model_id,temperature=0)

llm = ChatHuggingFace(llm=llm)

def mul(a:int, b:int):
    "mul takes 2 argument a and b, returns multiplication of those to numbers."
    return a*b
llm_with_tool = llm.bind_tools([mul])
#instead of state one we keep a list(so that we can know what steps have happend.)
class MessageState(TypedDict):
    messages: list[AnyMessage]


def toool(state:MessageState):

    return {"messages":[llm_with_tool.invoke(state["messages"])]}

builder = StateGraph(MessageState)
builder.add_node("toolcalling", toool)
builder.add_edge(START,"toolcalling")
builder.add_edge("toolcalling",END)

graph = builder.compile()



state = {"messages":"what is indias capital?"}
result = graph.invoke(state)
# print(result)
# print(state)

#to add like every stage(every human and models output) into state langgraph has MessageState.

from langgraph.graph import MessagesState

builder = StateGraph(MessagesState)

builder.add_node("toolcalling", toool)
builder.add_edge(START,"toolcalling")
builder.add_edge("toolcalling",END)

graph2 = builder.compile()

initial_message = [HumanMessage(content="what is 2 multiplied three?",name="jagdish")]
print(initial_message)
result = graph2.invoke({"messages":initial_message})
print(result)