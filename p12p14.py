from langchain_community.document_loaders import WebBaseLoader

loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
docs = loader.load()

loader = WebBaseLoader("https://lilianweng.github.io/posts/2024-02-05-human-data-quality/")
docs.extend(loader.load())
import uuid

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

chain = (
    {"doc": lambda x: x.page_content}
    | ChatPromptTemplate.from_template("Summarize the following document:\n\n{doc}")
    | ChatOpenAI(model="qwen2.5-14b-instruct",max_retries=0)
    | StrOutputParser()
)

summaries = chain.batch(docs, {"max_concurrency": 5})
from langchain.storage import InMemoryByteStore
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.retrievers.multi_vector import MultiVectorRetriever

# The vectorstore to use to index the child chunks

from sentence_transformers import SentenceTransformer
from langchain.embeddings.base import Embeddings
#from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
import os
# 加载本地模型（可以是任何Hugging Face 或 SentenceTransformers支持的模型）
model = SentenceTransformer(os.environ.get("bge"))  # 或者使用 Hugging Face 的模型名称

# 自定义嵌入函数
class LocalEmbedding(Embeddings):
    def embed_documents(self, texts):
        return model.encode(texts).tolist()

    def embed_query(self, query):
        query_embedding = model.encode([query])
        query_embedding = query_embedding.tolist()
        query_embedding = query_embedding[0]
        return query_embedding

# 使用自定义的嵌入函数
embedding_function = LocalEmbedding()


vectorstore = Chroma(collection_name="summaries",
                     embedding_function=embedding_function)

# The storage layer for the parent documents
store = InMemoryByteStore()
id_key = "doc_id"

# The retriever
retriever = MultiVectorRetriever(
    vectorstore=vectorstore,
    byte_store=store,
    id_key=id_key,
)
doc_ids = [str(uuid.uuid4()) for _ in docs]

# Docs linked to summaries
summary_docs = [
    Document(page_content=s, metadata={id_key: doc_ids[i]})
    for i, s in enumerate(summaries)
]

# Add
retriever.vectorstore.add_documents(summary_docs)
retriever.docstore.mset(list(zip(doc_ids, docs)))
query = "Memory in agents"
sub_docs = vectorstore.similarity_search(query,k=1)
print(sub_docs[0])
retrieved_docs = retriever.get_relevant_documents(query,n_results=1)
print(retrieved_docs[0].page_content[0:500])


print("###################################################Part 14################################################")
from ragatouille import RAGPretrainedModel
RAG = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")
import requests

def get_wikipedia_page(title: str):
    """
    Retrieve the full text content of a Wikipedia page.

    :param title: str - Title of the Wikipedia page.
    :return: str - Full text content of the page as raw string.
    """
    # Wikipedia API endpoint
    URL = "https://en.wikipedia.org/w/api.php"

    # Parameters for the API request
    params = {
        "action": "query",
        "format": "json",
        "titles": title,
        "prop": "extracts",
        "explaintext": True,
    }

    # Custom User-Agent header to comply with Wikipedia's best practices
    headers = {"User-Agent": "RAGatouille_tutorial/0.0.1 (ben@clavie.eu)"}

    response = requests.get(URL, params=params, headers=headers)
    data = response.json()

    # Extracting page content
    page = next(iter(data["query"]["pages"].values()))
    return page["extract"] if "extract" in page else None

full_document = get_wikipedia_page("Hayao_Miyazaki")
RAG.index(
    collection=[full_document],
    index_name="Miyazaki-123",
    max_document_length=180,
    split_documents=True,
)
results = RAG.search(query="What animation studio did Miyazaki found?", k=3)
print(results)
retriever = RAG.as_langchain_retriever(k=3)
print(retriever.invoke("What animation studio did Miyazaki found?"))