from langchain_huggingface import ChatHuggingFace,  HuggingFaceEmbeddings, HuggingFaceEndpoint, HuggingFaceEndpointEmbeddings

from langchain_text_splitters import RecursiveCharacterTextSplitter
import numpy as np
import os
from langchain_classic import hub


embedder = HuggingFaceEndpointEmbeddings(model = "sentence-transformers/all-mpnet-base-v2",
                                         task="feature-extraction",
                                         huggingfacehub_api_token=os.environ.get("HF_TOKEN"))

def getEmbedder(text):
    return embedder.embed_query(text)

def cosineSimilarity(a,b):
    dotP = np.dot(a,b)
    norma = np.linalg.norm(a)
    normb = np.linalg.norm(b)
    return (dotP/(norma*normb))

import bs4
from langchain_community.document_loaders import WebBaseLoader
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
blog_docs = loader.load()

text_spiliter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=300,chunk_overlap=50)
# print(blog_docs)
splits = text_spiliter.split_documents(blog_docs)
# print(splits)

from langchain_community.vectorstores import Chroma
vectorstore = Chroma.from_documents(documents=splits, 
                                    embedding=embedder)
retriever = vectorstore.as_retriever(search_kwargs={"k":1})
prompt = "what is Task Decomposition?"
system_prompt = "Given the context answer following."
docs = retriever.invoke(prompt)
print(len(docs))
# print(docs)

# from langchain.chat_models import Chat
model_id = "meta-llama/Llama-3.2-3B-Instruct"
llm = HuggingFaceEndpoint(repo_id=model_id,temperature=0)
llm = ChatHuggingFace(llm=llm)

from langchain_core.prompts import ChatPromptTemplate

template = """Answer the question based only on the following context:
{context}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)
print(prompt)
chain = prompt | llm
# print(chain.invoke({"context":docs,"question":"What is Task Decomposition?"}))
# print(llm.invoke("Context: "+docs[0].page_content+"\n"+system_prompt+prompt))
# from  langchain import hub


prompt_hub_rag = hub.pull("rlm/rag-prompt")
chain = prompt_hub_rag | llm

# print(prompt_hub_rag)
print(chain.invoke({"context":docs,"question":"What is computer?"}))