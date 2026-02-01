from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

from langgraph.graph import StateGraph, START, END, MessagesState

from langgraph.prebuilt import tools_condition, ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langchain.messages import HumanMessage


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
builder.add_edge("tools","toolcalling")
memory = MemorySaver()
graph = builder.compile(checkpointer=memory)
config = {"configurable": {"thread_id": "1"}}

# Specify an input
messages = [HumanMessage(content="Add 3 and 4.")]

# Run
messages = graph.invoke({"messages": messages},config)

print(config)

# answer = graph.invoke({"messages":["what is 3 add 2 and then multiply 4?"]})
for m in messages['messages']:
    m.pretty_print()
# print(answer)
# config = {"configurable": {"thread_id": "2"}}
messages = graph.invoke({"messages":["that multiplied by 2?"]},config=config)

for m in messages['messages']:
    m.pretty_print()