from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

model_id = "meta-llama/Llama-3.3-70B-Instruct"

llm = HuggingFaceEndpoint(repo_id=model_id, temperature=0.7)
llm = ChatHuggingFace(llm=llm)


def multiply(a:int,b:int) -> int:
    """Multiplies a and b.

    Args:
        a: first int
        b: second int
    """
    # print("multiply")
    return a*b
def add(a:int, b:int):
    """adds a and b.

    Args:
        a: first int
        b: second int
    """
    print("aadding")
    return a + b
def sub(a:int, b:int):
    """subtracts a and b.

    Args:
        a: first int
        b: second int
    """
    print("aadding")
    return a - b

llm_with_tool = llm.bind_tools([multiply,add, sub])

result = llm_with_tool.invoke("I had 3 apples, I gave 2 to my brother. how many apples do I have now?")
print(result)