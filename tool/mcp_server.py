from fastmcp import FastMCP

from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
# from langchain.retrievers.document_compressors import FlashrankRerank
from langchain_classic.retrievers.document_compressors import FlashrankRerank
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

import yaml
from pathlib import Path
# import rag

# Create a server instance
mcp = FastMCP(name="MCPServer")

# Initilize
global vectordb_path
global MODEL
global API_KEY
with open("./config/config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)
    vectordb_path = config["CHROMA_PATH"]
    MODEL = config["LLM"]["MODEL"]
    API_KEY = config["LLM"]["API_KEY"]

## === Dev-test ===
@mcp.tool
def get_secret()->str:
    """
    Get a secret key
    """
    sk = "13345678"

    return sk

## === Manage vector DB collection ===
def get_all_collection()->str:
    """
    Get all of collection in chroma DB
    [Dev] need modify, get information from reading collections meta-info
    Agent should be able to get meta-info of collections
    Return: a str of list collection name and description
    """
    # vector_store = Chroma(persist_directory=vectordb_path)
    # client = vector_store._client
    # collection_names = [col.name for col in client.list_collections()]

    with open("./collection_info.yaml", "r", encoding="utf-8") as f:
        collection_info = yaml.safe_load(f)
        info_str = yaml.dump(collection_info, sort_keys=False, allow_unicode=True)

    return info_str

def update_collections():
    """
    Update collection meta-info
    USER->add/delete/update collections->sys:update_collection meta-info
    """

def retrieve_collection(query)->list:
    """
    LLM Retrieve from collection meta-info to get a list of collections that relevant to query
    Return: List of relevant collection name
    """
    prompt = PromptTemplate(
        template="""Please retrieve from the given collection to select the query-relevant collection.
        The output format should be like：['apple', 'banana']\n
        Query: {query}
        Collections: {collection}\n
        Your Response:
        """,
        input_variables=["query", "collection"]
    )
    # llm_client = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key="AIzaSyASSpK37mLNMmjiHESB1QfvTqmAgwy6770")
    llm_client = ChatGoogleGenerativeAI(model=MODEL, google_api_key=API_KEY)
    parser = JsonOutputParser()

    chain = (prompt | llm_client | parser)

    res = chain.invoke({"query": query, "collection":get_all_collection()})

    return res





def get_document(collection_name):
    """
    Docstring for get_document
    """
    collections = get_all_collection()
    target_collection = []
    if collection_name.lower()=="all":
        target_collection = get_all_collection()
    elif collection_name in collections:
        target_collection = [collection_name]
    else:
        raise ValueError("Unknow Collection")
    
    for collect in target_collection:
        vector_store = Chroma(
        collection_name=collect,
        persist_directory=vectordb_path,
        )
        documents = set([meta["source"] for meta in vector_store.get()["metadatas"]])
    
    return documents

def delete_document():
    """
    Docstring for delete_document
    """

## === RAG Function===
def rag_upload():
    """
    USER upload doc (upload->embedding->store)
    """

def rag_retrieval(collections:list, query)->str:
    """
    Retrieve relevant information from given collections and re-rank result.
    """
    try:
        # Retrieve from the given collections
        retrieval = []
        for collection in collections:
            vector_store = Chroma(
                collection_name=collection,
                persist_directory=vectordb_path,
            )
            retriever = vector_store.as_retriever(search_kwargs={"k": 10})  # RAG top-K=10
            result = retriever.invoke(query)
            retrieval.extend(result)

        # Re-Ranking from given retrieved results
        compressor = FlashrankRerank(top_n=5)
        rerank_res = compressor.compress_documents(documents=retrieval, query=query)
    
    except Exception as e:
        return f"Retrieval Error: {e}"

    return rerank_res


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
    mcp.run(transport="http", host="127.0.0.1", port=7007)