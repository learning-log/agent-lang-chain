from langchain.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph,START, END
from langgraph.graph import MessagesState
from typing import TypedDict
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.messages import RemoveMessage

model_id = "mistralai/Mistral-7B-Instruct-v0.3"

llm = HuggingFaceEndpoint(repo_id=model_id, temperature=0.7)
llm = ChatHuggingFace(llm=llm)
def llm_call(state:MessagesState):
    return {"messages":llm.invoke(state["messages"])}
builder = StateGraph(MessagesState)
builder.add_node("llm_call",llm_call)
builder.add_edge(START,"llm_call")
builder.add_edge("llm_call",END)
graph = builder.compile()

print(graph.invoke({"messages":"hello!"}))

#if you are having so many turns of conversion then its difficult for model to process these many calculations. so remove message can be used.
def filter(state:MessagesState):
    if len(state["messages"])<=2:
        print("not_removed")
        return state
    else:
        removed_message = [RemoveMessage(id= m.id) for m in state["messages"]]
        return {"message":removed_message}

builder = StateGraph(MessagesState)
builder.add_node("llm_call",llm_call)
builder.add_node("filter",filter)
builder.add_edge("filter","llm_call")
builder.add_edge(START,"filter")
builder.add_edge("llm_call",END)
graph = builder.compile()

message = graph.invoke({"messages":"hello"})
message["messages"].append(HumanMessage("I am good. what is canadian currency?"))

message = graph.invoke({"messages":message["messages"]})

message["messages"].append(HumanMessage("does this country has strict gun laws?"))
message = graph.invoke({"messages":message["messages"]})

# for m in message["messages"]:
#     m.pretty_print()