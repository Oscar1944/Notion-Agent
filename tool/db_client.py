import chromadb
import yaml
import numpy as np
import uuid
import os
import sys
from chromadb.utils import embedding_functions
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# DB setting
global DB_PATH
global COLLECTION

# Chunking Setting
global CHUNK_SIZE
global CHUNK_OVERLAP

# DB Embedding Setting (For initializing a new collection)
global EMBEDDING_MODEL
global SPACE
global DIM

# DB Searching Setting
global SEARCH_THRESHOLD
global TOP_K
global RERANK_TOP_N

with open(r"C:\Users\USER\Desktop\VS Code file\RAG-Agent\Notion-Agent\config\chroma_config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)
    if config["DATABASE"]!="CHROMA":
        raise ValueError("Database only support Chroma, please make sure applying Chroma DB.")
    
    DB_PATH = config["DB_PATH"]
    COLLECTION = config["COLLECTION"]

    CHUNK_SIZE = config["CHUNK_SIZE"]
    CHUNK_OVERLAP = config["CHUNK_OVERLAP"]

    EMBEDDING_MODEL = config["EMBEDDING_MODEL"]
    SPACE = config["SPACE"]
    DIM = config["DIM"]

    SEARCH_THRESHOLD = config["SEARCH_THRESHOLD"]
    TOP_K = config["TOP_K"]
    RERANK_TOP_N = config["RERANK_TOP_N"]


class ChromaDB_Client:
    def __init__(self, DB_PATH, 
                 COLLECTION_NAME, 
                 CHUNK_SIZE=CHUNK_SIZE,
                 CHUNK_OVERLAP=CHUNK_OVERLAP,
                 EMBEDDING_MODEL=EMBEDDING_MODEL, 
                 EMBED_SPACE=SPACE, 
                 EMBED_DIM=DIM,
                 SEARCH_THRESHOLD=SEARCH_THRESHOLD,
                 TOP_K=TOP_K,
                 RERANK_TOP_N=RERANK_TOP_N
                 ):
    #    DB Setting
       self.DB_PATH = DB_PATH
       self.COLLECTION_NAME = COLLECTION_NAME

    #    Chunking Setting
       self.CHUNK_SIZE = CHUNK_SIZE
       self.CHUNK_OVERLAP = CHUNK_OVERLAP

    #    Embedding Setting
       self.EMBEDDING_MODEL = EMBEDDING_MODEL
       self.EMBED_SPACE = EMBED_SPACE
       self.EMBED_DIM = EMBED_DIM
       self.EMBED_FUNCTION = self.get_embedding_func()  # Langchain Embedding function

    #    DB/Collection Client Setting
       self.DB_CLIENT = self.get_db_client()
       self.COLLECTION_CLIENT = self.get_collection_client()

    #    Searching Setting
       self.SEARCH_THRESHOLD=SEARCH_THRESHOLD
       self.TOP_K=TOP_K
       self.RERANK_TOP_N=RERANK_TOP_N

       print("✅ DB Collection has initialized Successfuly")

    def get_embedding_func(self):
        """
        Get embedding function based on assigned embedding model name.
        """
        if self.EMBEDDING_MODEL=="all-MiniLM-L6-v2":
            embed_func = embedding_functions.DefaultEmbeddingFunction()  # Chroma DB applies 'all-MiniLM-L6-v2' as default embedding.
        else:
            embed_func = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=self.EMBEDDING_MODEL)
        return embed_func
    
    def get_db_client(self):
        """
        Get a DB Client based on DB path
        """
        db_client = chromadb.PersistentClient(path=self.DB_PATH)
        return db_client

    def get_collection_client(self):
        """
        Get a Collection Client based on collection name
        """
        existing_collections = [c.name for c in self.DB_CLIENT.list_collections()]
        if self.COLLECTION_NAME not in existing_collections:
            print(f"⚠️   Collection {self.COLLECTION_NAME} not found.")
            print(f"""
                  Do you want to buid collection {self.COLLECTION_NAME} (y/n) ?
                  *** Collection Details ***
                  - Collection Name: {self.COLLECTION_NAME}
                  - Embedding Model: {self.EMBEDDING_MODEL}
                  - Embedding Dimension: {self.EMBED_DIM}
                  """)
            user_confirm = input()
            if user_confirm.lower()=="y" or user_confirm.lower()=="yes":
                pass
            else:
                sys.exit()
        
        collection_client = self.DB_CLIENT.get_or_create_collection(
                        name=self.COLLECTION_NAME,
                        embedding_function=self.EMBED_FUNCTION,
                        metadata={
                            "hnsw:space": self.EMBED_SPACE,       # embbedding space (cosine, l2, ip)
                            "model_name": self.EMBEDDING_MODEL,     # embedding model name
                            "dimension": self.EMBED_DIM               # embedding model dim
                        }
                    )
        return collection_client

    def retrieve(self, query:str):
        """
        Get a retrieval results for DB based on given query. (Retrieve->filter-by-threshold)
        Input: (str) query
        Return: List[Dict] A list of chunks and its content, metadata, source
        """
        results = self.COLLECTION_CLIENT.query(
            query_texts=[query],
            n_results=self.TOP_K,
            include=["documents", "metadatas", "distances"] # 確保抓取內容與標籤
        )
        final_docs = []
        if results['documents']:
            for i in range(len(results['documents'][0])):
                doc = results['documents'][0][i]
                metadata = results['metadatas'][0][i]
                distance = results['distances'][0][i]
                
                # Filter relevant chunks based on embed space type threshold
                if distance <= self.SEARCH_THRESHOLD:
                    final_docs.append({
                        "content": doc,
                        "metadata": metadata,
                        "score": distance
                    })
        return final_docs

    def get_all_document(self):
        """
        Get all documens in the colelction.
        Return: List[str] A list document names
        """
        metadatas = self.COLLECTION_CLIENT.get(include=["metadatas"])  # Get metadata of all chunks
        all_meta = metadatas["metadatas"]

        docs = set()
        for meta in all_meta:
            doc_name = os.path.basename(meta.get("source", "unknown"))
            if doc_name not in docs:
                docs.add(doc_name)
        return list(docs)

    def add_document(self, upload_file_path:str):
        """
        Add a file into DB. (check file existed->chunking->assign UUID->embedding->store vector) 
        """
        try:
            all_docs = set(self.get_all_document())
            filename = os.path.basename(upload_file_path)
            if filename in all_docs:
                print("Filename conflict detected in DB. Please make sure file is not duplicated or has same filename.")
                sys.exit()

            # Load file & Chunking
            loader = PyPDFLoader(upload_file_path)
            data = loader.load()

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
            chunks = text_splitter.split_documents(data)

            # Assign UUID
            documents_list = []
            metadatas_list = []
            ids_list = []
            for chunk in chunks:
                documents_list.append(chunk.page_content)
                metadatas_list.append(chunk.metadata)
                ids_list.append(str(uuid.uuid4()))  # Generate ID for each chunks

            # Store into DB
            self.COLLECTION_CLIENT.add(
                ids=ids_list,
                documents=documents_list,
                metadatas=metadatas_list
            )
        except Exception as e:
            print(e)
            raise ValueError(f"Upload file to DB has error")
        print("✅ File upload successdfuly")
    
    def delete_document(self, delete_file_path:str):
        """
        Delete a file from DB collection
        """
        try:
            self.COLLECTION_CLIENT.delete(where={"source": delete_file_path})
        except Exception as e:
            print(e)
            raise ValueError("Error occur when deleting files in DB")
        print("✅ File delete successfuly")
    

# c = ChromaDB_Client(DB_PATH, COLLECTION, SEARCH_THRESHOLD=100)
# f = c.get_all_document()
# print(f)

# d = c.retrieve("what is AI")
# print(d)

# c.add_document(r"C:\Users\USER\Desktop\VS Code file\RAG-Agent\Spark_env_setting_practice.pdf")
# c.delete_document(r"C:\Users\USER\Desktop\VS Code file\RAG-Agent\Spark_env_setting_practice.pdf")
# f = c.get_all_document()
# print(f)

