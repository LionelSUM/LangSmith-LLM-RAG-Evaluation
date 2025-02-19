# Documents
from overview import create_embeddings_faiss, vector_db_path

question = "What kinds of pets do I like?"
document = "My favorite pet is a cat."

import tiktoken

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

num_tokens_from_string(question, "cl100k_base")

from langchain_openai import ChatOpenAI
embd = ChatOpenAI(model_name="qwen2.5-14b-instruct", temperature=0)
query_result = embd.invoke([question])
document_result = embd.invoke([document])
print(query_result)
print(document_result)

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

def cosine_similarity(vec1, vec2):
    vec1 = vec1.flatten()
    vec2 = vec2.flatten()

    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)

query_text = query_result.content
document_text = document_result.content

# 使用 TF-IDF 转换文本
vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform([query_text, document_text])

# 获取向量
query_vector = vectors[0].toarray()
document_vector = vectors[1].toarray()

similarity = cosine_similarity(query_vector, document_vector)
print("Cosine Similarity:", similarity)

#### INDEXING ####
# Load blog
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

# Split
from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=300,
    chunk_overlap=50)

# Make splits
splits = text_splitter.split_documents(blog_docs)

# Index
from langchain_openai import OpenAIEmbeddings
vectorstore = create_embeddings_faiss(
                            vector_db_path=vector_db_path,
                            embedding_name="bge",
                            chunks=splits
                        )

retriever = vectorstore.as_retriever(search_kwargs={"k": 1})

docs = retriever.invoke("What is Task Decomposition?")
len(docs)

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

# Prompt
template = """Answer the question based only on the following context:
{context}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

# LLM
llm = ChatOpenAI(model_name="qwen2.5-14b-instruct", temperature=0)
# Chain
chain = prompt | llm
# Run
chain.invoke({"context":docs,"question":"What is Task Decomposition?"})
from langchain import hub
prompt_hub_rag = hub.pull("rlm/rag-prompt")

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

print(rag_chain.invoke("What is Task Decomposition?"))