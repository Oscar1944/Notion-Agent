from fastmcp import FastMCP
import os
import chromadb
from db_client import ChromaDB_Client
from chromadb.utils import embedding_functions
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_classic.retrievers.document_compressors import FlashrankRerank
from flashrank import Ranker
FlashrankRerank.model_rebuild()
FlashrankRerank.model_rebuild(_types_namespace={"Ranker": Ranker})
from flashrank import Ranker
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

# DB Searching Setting
global SEARCH_THRESHOLD
global TOP_K
global RERANK_TOP_N

# Customized DB instance
global ChromaDB  

with open("./config/chroma_config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)
    if config["DATABASE"]!="CHROMA":
        raise ValueError("Database only support Chroma, please make sure applying Chroma DB.")
    
    DB_PATH = config["DB_PATH"]
    COLLECTION = config["COLLECTION"]

    # SEARCH_THRESHOLD = config["SEARCH_THRESHOLD"]
    SEARCH_THRESHOLD = 100
    TOP_K = config["TOP_K"]
    RERANK_TOP_N = config["RERANK_TOP_N"]

    ChromaDB = ChromaDB_Client(DB_PATH, COLLECTION, SEARCH_THRESHOLD=SEARCH_THRESHOLD, TOP_K=TOP_K)


## === Dev-test ===
@mcp.tool()
def get_secret()->str:
    """
    Get a secret key
    """
    sk = "13345678"

    return sk

# === RAG ===
# @mcp.tool()
def rag_retrieval(query:str)->str:
    """
    Search the database for relevant information based on the user's query.
    Returns a string containing the relevant snippets or an empty message if nothing found.
    Input (str): query
    Return (str): Relevant content or empty if none.
    """
    # Retrieve relevant information as reference from given collections and re-rank retrieved result.
    final_retrieval = ""
    try:
        # Retrieve chunks from the DB collections , Get List[Dict]->Dict(content, metadata(dict), score)
        retrieval_result = ChromaDB.retrieve(query)

        # Convert ChromaDB spec into Langchain Chroma spec, List[Langchain Document obj]
        langchain_retrieval_result = []
        for result in retrieval_result:
            langchain_retrieval_result.append(
                Document(page_content=result["content"], metadata=result["metadata"])
            )

        # Re-Ranking from given retrieved results
        if langchain_retrieval_result:
            # Initilize Re-Ranker & Re-Ranking
            compressor = FlashrankRerank(top_n=RERANK_TOP_N)
            rerank_retrieval_result = compressor.compress_documents(documents=langchain_retrieval_result, query=query)

            # Combine re-ranked result into a single string, () is not a tuple is used to combine str
            final_retrieval = [
                (
                    f"Source: {os.path.basename(rerank_result.metadata.get('source', 'unknown'))}\n"
                    f"Content: {rerank_result.page_content}"
                )
                for rerank_result in rerank_retrieval_result
            ]
            final_retrieval = "\n\n".join(final_retrieval)
    
    except Exception as e:
        final_retrieval = f"Retrieval Error: {e}"

    return final_retrieval


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
    # mcp.run(transport="http", host="127.0.0.1", port=7007)

    # dev-test
    res = rag_retrieval("地道")

    print(res)