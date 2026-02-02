from langchain.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph,START, END
from langgraph.graph import MessagesState
from typing import TypedDict
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.messages import RemoveMessage
from langgraph.checkpoint.memory import MemorySaver

model_id = "meta-llama/Llama-3.2-3B-Instruct"

llm = HuggingFaceEndpoint(repo_id=model_id, temperature=0.7)
llm = ChatHuggingFace(llm=llm)
class CMessagesState(MessagesState):
    summary: str
def llm_call(state:CMessagesState):
    print("########")
    print(state)
    summary = state.get("summary","")
    
    if summary!="":
        message = [SystemMessage(content=summary)] + state["messages"]
    else:
        message = state["messages"]

    return {"messages":llm.invoke(message)}

def summarization(state:CMessagesState):
    summary = state.get("summary","")

    if summary!="":
        message = f"This is the summary till now summary: {summary}. Above are the new messages, generate the complete summary."
    else:
        message = f"Generate the summary of the above conversation."
    message = state["messages"]+[HumanMessage(content=message)]
    summary = llm.invoke(message)

    delete_messages = [RemoveMessage(id = m.id) for m in state["messages"][:-2]]

    return {"messages":delete_messages,"summary":summary.content}

def decider(state:CMessagesState):

    if len(state["messages"])<6:
        return END
    else:
        return "summarization"

builder =  StateGraph(CMessagesState)
builder.add_node("llm_call", llm_call)
builder.add_node("summarization",summarization)
builder.add_edge(START,"llm_call")
builder.add_conditional_edges("llm_call",decider)
builder.add_edge("summarization",END)
memory = MemorySaver()

graph = builder.compile(checkpointer=memory)
config = {"configurable": {"thread_id": "1"}}
graph.invoke({"messages":"hello"},config=config)

graph.invoke({"messages": "tell me about what is the capital of canada?"},config=config)

graph.invoke({"messages":"how is the whether in that city in winters?"},config=config)

print(graph.invoke({"messages":"what type of clothes should we wear in that whether"},config))
