from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough


class RAGPipeine:
    def __init__(self, collection_name, chunk_size=500, chunk_overlap=50):
        self.collection_name = collection_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.retriever = None
    
            
    def upload(self, pdf_path):
        """
        Upload a PDF document, process it, and store it in the vector database.
        """
        try:
            # Load PDF document
            loader = PyPDFLoader(pdf_path)
            data = loader.load()

            # Chunking
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            chunks = text_splitter.split_documents(data)

            # Embedding & Store vector
            vector_store = Chroma(
                collection_name=self.collection_name,  # DB collection name
                persist_directory="./chroma_db",   # path to store vector
                # embedding_function=""   # Default: MiniLM-L6-v2 embedding
            )
            vector_store.add_documents(documents=chunks)

            # Initilize as retriever based-on curretn DB
            self.retriever = vector_store.as_retriever()

        except Exception as e:
            return f"Upload Error: {e}"
        
        return "Upload successful!"

    def retrieval(self, query)->list:
        """
        Retrieve relevant information from Vector DB based on the query and generate an answer using the LLM.
        """
        try:
            result = self.retriever.invoke(query)
        except Exception as e:
            return f"Retrieval Error: {e}"

        return result

    def query(self, query, client):
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
                    "context": self.retriever | (lambda docs: "\n".join([doc.page_content for doc in docs])),
                    "question": RunnablePassthrough()
                }
                | prompt_template
                | client
            )
        res = qa_chain.invoke(query)

        return res
        

    



