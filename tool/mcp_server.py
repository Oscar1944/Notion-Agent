from fastmcp import FastMCP

import chromadb
from chromadb.utils import embedding_functions

from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_classic.retrievers.document_compressors import FlashrankRerank
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

import yaml
import sys

# Create a server instance
mcp = FastMCP(name="MCPServer")

# Initilize LLM/API
global MODEL
global API_KEY
with open("./config/config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)
    MODEL = config["LLM"]["MODEL"]
    API_KEY = config["LLM"]["API_KEY"]

# DB setting
global DB_PATH
global COLLECTION
global COLLECTION_META  # Store details of current collection

# DB Embedding Setting (For initializing a new collection)
global EMBEDDING_MODEL
global SPACE
global DIM

# DB Searching Setting
global SEARCH_TYPE
global SEARCH_THRESHOLD
global TOP_K
global RERANK_TOP_N

with open("./config/chroma_config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)
    if config["DATABASE"]!="CHROMA":
        raise ValueError("Database only support Chroma, please make sure applying Chroma DB.")
    
    DB_PATH = config["DB_PATH"]
    COLLECTION = config["COLLECTION"]

    EMBEDDING_MODEL = config["EMBEDDING_MODEL"]
    SPACE = config["SPACE"]
    DIM = config["DIM"]

    SEARCH_TYPE = config["SEARCH_TYPE"]
    SEARCH_THRESHOLD = config["SEARCH_THRESHOLD"]
    TOP_K = config["TOP_K"]
    RERANK_TOP_N = config["RERANK_TOP_N"]


## === Dev-test ===
@mcp.tool
def get_secret()->str:
    """
    Get a secret key
    """
    sk = "13345678"

    return sk

## === RAG Function===
def init_collection_meta(db_path, collection_name):
    """
    Get meta-data of a collection.
    """
    global COLLECTION_META
    embed_func = None  # Embedding function obj
    collection_meta = {}  # collection details
    
    # DB Client
    client = chromadb.PersistentClient(path=db_path) 

    # Check Embedding model
    if EMBEDDING_MODEL=="all-MiniLM-L6-v2":
        embed_func = embedding_functions.DefaultEmbeddingFunction()  # Chroma DB applies 'all-MiniLM-L6-v2' as default embedding.
    else:
        embed_func = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)

    # Check if collection existed
    existing_collections = [c.name for c in client.list_collections()]
    if collection_name in existing_collections:
        pass
    else:
        print(f"⚠️ Collection {collection_name} not found. Do you want to buid collection {collection_name} (y/n) ?")
        user_confirm = input()
        if user_confirm.lower()=="y" or user_confirm.lower()=="yes":
            # Build collection
            collection = client.get_or_create_collection(
                name=collection_name,
                embedding_function=embed_func,
                metadata={
                    "hnsw:space": SPACE,       # embbedding space (cosine, l2, ip)
                    "model_name": EMBEDDING_MODEL,# model name
                    "dimension": DIM               # model dim
                }
            )
        else:
            print("SYSTEM STOP")
            sys.exit()
    
    # === Collection Details ===
    col = client.get_collection(name=collection_name)
    meta = col.metadata or {}
    collection_meta["dim"] = meta.get("dimension")  # Get embedding dim
    collection_meta["space"] = meta.get("hnsw:space")  # Get searching metrics
    collection_meta["embedding"] = meta.get("model_name")  # Get embedding model name
    collection_meta["embed_func"] = embed_func
    COLLECTION_META = collection_meta

    print("✅ DB Collection has initialized Successfuly")

# @mcp.tool
def rag_retrieval(query)->str:
    """
    Retrieve relevant information as reference from given collections and re-rank retrieved result.
    This should provide reference to answer the question.
    Input: the given query
    Return: A list of references that relevant to query.
    """
    # The design of rag_retrieval() support multi-collection
    try:
        # Retrieve from the given collections
        vector_store = Chroma(
            persist_directory=DB_PATH,
            collection_name=COLLECTION,
            embedding_function=COLLECTION_META["embed_func"]
        )
        retriever = vector_store.as_retriever(
            search_type=SEARCH_TYPE, 
            search_kwargs={"k": TOP_K, "score_threshold": SEARCH_THRESHOLD}    # RAG top-K=10
        )  
        retrieval = retriever.invoke(query)

        # Re-Ranking from given retrieved results
        if retrieval:
            compressor = FlashrankRerank(top_n=RERANK_TOP_N)
            retrieval = compressor.compress_documents(documents=retrieval, query=query)  # Re-Ranking
    
    except Exception as e:
        raise ValueError(f"Retrieval Error: {e}")

    return retrieval


def rag_query(chroma_client, llm_client, query):
    """
    Answer a given question based on the current collection
    """
    prompt_template = ChatPromptTemplate.from_template(
    """Answer the following question based on the provided context:

    Context: {context}

    Question: {question}

    Answer:
    """
    )
    qa_chain = (
            {
                "context": rag_retrieval(chroma_client, query) | (lambda docs: "\n".join([doc.page_content for doc in docs])),
                "question": RunnablePassthrough()
            }
            | prompt_template
            | llm_client
        )
    res = qa_chain.invoke(query)

    return res

## === Notion ===


if __name__=="__main__":
    init_collection_meta(DB_PATH, COLLECTION)
    # mcp.run(transport="http", host="127.0.0.1", port=7007)

    # dev-test
    res = rag_retrieval("what is overfitting")

    print(res)