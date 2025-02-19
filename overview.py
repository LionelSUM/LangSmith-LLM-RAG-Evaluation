import bs4
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def get_embedding(embedding_name):
    embedding_path = os.environ.get(embedding_name)  # 使用环境变量获取路径
    if not embedding_path:
         raise ValueError("bge 环境变量未设置或路径无效！")
    model_kwargs = {'device': 'cpu'}
    return HuggingFaceEmbeddings(model_name=embedding_path, model_kwargs=model_kwargs)

def create_embeddings_faiss(vector_db_path, embedding_name, chunks):
    embeddings = get_embedding(embedding_name)
    db = FAISS.from_documents(chunks, embeddings)

    if not os.path.isdir(vector_db_path):
        os.mkdir(vector_db_path)

    db.save_local(folder_path=vector_db_path)
    return db


def load_embeddings_faiss(vector_db_path, embedding_name):
    embeddings = get_embedding(embedding_name)
    db = FAISS.load_local(vector_db_path, embeddings, allow_dangerous_deserialization=True)
    return db

BASE_DIR = os.path.dirname(__file__)
vector_db_path = os.path.join(BASE_DIR, "vector_db")

#### INDEXING ####

# Load Documents
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
docs = loader.load()

# Split
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

# Embed
vectorstore = create_embeddings_faiss(
                            vector_db_path=vector_db_path,
                            embedding_name="bge",
                            chunks=splits
                        )

retriever = vectorstore.as_retriever()

#### RETRIEVAL and GENERATION ####

# Prompt
prompt = hub.pull("rlm/rag-prompt")

# LLM
llm = ChatOpenAI(model_name="qwen2.5-14b-instruct", temperature=0)

# Post-processing
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Chain
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Question
print(rag_chain.invoke("What is Task Decomposition?"))