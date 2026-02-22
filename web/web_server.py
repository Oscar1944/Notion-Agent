import os
import sys
import json
import uuid

# Add parent directory to path so we can import sibling modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import chromadb
from tool.db_client import ChromaDB_Client
from langchain_chroma import Chroma
from chromadb.utils import embedding_functions
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Response
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import httpx
import asyncio
from pydantic import BaseModel
from agent.agent import Agent

try:
    import yaml
except Exception:
    yaml = None

from langchain_mcp_adapters.client import MultiServerMCPClient  
from langchain_google_genai import ChatGoogleGenerativeAI

HERE = os.path.dirname(__file__)
STATIC_DIR = HERE
PARENT_DIR = os.path.dirname(HERE)
UPLOAD_DIR = os.path.join(PARENT_DIR, 'uploads')
os.makedirs(UPLOAD_DIR, exist_ok=True)

# === FastAPI Website ===
app = FastAPI()

# === CORS Configuration ===
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (adjust for production)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BACKEND = os.environ.get('BACKEND_URL','http://localhost:8000')

# === Chroma DB & Collection ===
# DB Setting
global DB_PATH
global COLLECTION

# Chunking Setting
global CHUNK_SIZE
global CHUNK_OVERLAP

# Customized DB instance
global ChromaDB  

with open(r"C:\Users\USER\Desktop\VS Code file\RAG-Agent\Notion-Agent\config\chroma_config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)
    if config["DATABASE"]!="CHROMA":
        raise ValueError("Database only support Chroma, please make sure applying Chroma DB.")
    # DB_PATH = config["DB_PATH"]
    DB_PATH = r"C:\Users\USER\Desktop\VS Code file\RAG-Agent\Notion-Agent\chroma_db"
    COLLECTION = config["COLLECTION"]
    CHUNK_SIZE = config["CHUNK_SIZE"]
    CHUNK_OVERLAP = config["CHUNK_OVERLAP"]
    ChromaDB = ChromaDB_Client(DB_PATH, COLLECTION, CHUNK_SIZE=CHUNK_SIZE, CHUNK_OVERLAP=CHUNK_OVERLAP)

# === Agent (LLM, MCP) ===
global AGENT
global toolkit

# LLM_Client = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=json.load("../test_config.json").get("key"))
LLM_Client = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key="-8")
MCP_Client = MultiServerMCPClient(
        {
            "Tools": {
                "transport": "http",  # HTTP-based remote server
                "url": "http://localhost:7007/mcp"  # URL to mcp server
            }
        }
    )


# === Request ===
class ChatRequest(BaseModel):
    query: str

# === Startup Event ===
@app.on_event("startup")
async def startup_event():
    global toolkit, AGENT
    toolkit = await MCP_Client.get_tools()
    AGENT = Agent(LLM_Client, toolkit)
    print("✓ Agent initialized successfully")

#  === CRUD Dossier ===
@app.get('/api/dossier/files')
def list_dossier_files():
    """
    API to Get all of files (filename, file-size)
    """
    files = []
    try:
        for fname in os.listdir(UPLOAD_DIR):
            full = os.path.join(UPLOAD_DIR, fname)
            if os.path.isfile(full):
                files.append({'name': fname, 'size': os.path.getsize(full)})
    except Exception:
        files = []
    return {'files': files}


@app.post('/api/dossier/files')
async def upload_dossier_file(file: UploadFile = File(...)):
    """
    API to upload file to Dossier
    """
    safe_name = os.path.basename(file.filename)
    dest = os.path.join(UPLOAD_DIR, safe_name)  # File uploaded path
    try:
        contents = await file.read()
        with open(dest, 'wb') as f:
            f.write(contents)  # save file into uploads/
            # db_upload(dest)  # save file as vector into DB
            ChromaDB.add_document(dest)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {'name': safe_name, 'size': os.path.getsize(dest)}


@app.delete('/api/dossier/files/{filename}')
def delete_dossier_file(filename: str):
    """
    API to delete file from Dossier
    """
    safe = os.path.basename(filename)
    path = os.path.join(UPLOAD_DIR, safe) # File uploaded path
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail='File not found')
    try:
        os.remove(path)  # remove file from uploaded path uploads/
        # db_delete(path)  # remove file from DB
        ChromaDB.delete_document(path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return Response(status_code=204)


# === Send message to Agent and get Agent response ===
@app.post('/api/chat')
async def agent_response(request: ChatRequest):
    query = request.query
    try:
        response = await AGENT.chat(query)
    except Exception as e:
        print(e)
        raise HTTPException(status_code=505, detail='Agent has error during response genreation') 
    
    return JSONResponse(response)


# === Root (Web init redering) ===
@app.get('/', include_in_schema=False)
async def root():
    return FileResponse(os.path.join(STATIC_DIR, 'index.html'))


@app.get('/{file_path:path}', include_in_schema=False)
async def serve_static(file_path: str):
    file_full_path = os.path.join(STATIC_DIR, file_path)
    if os.path.isfile(file_full_path):
        return FileResponse(file_full_path)
    # Try index.html for single page apps
    if os.path.isfile(os.path.join(STATIC_DIR, 'index.html')):
        return FileResponse(os.path.join(STATIC_DIR, 'index.html'))
    raise HTTPException(status_code=404, detail='File not found')


@app.api_route('/api/{path:path}', methods=['GET','POST','PUT','DELETE','PATCH'], include_in_schema=False)
async def proxy(request: Request, path: str):
    # Simple proxy to forward requests to BACKEND
    target = BACKEND.rstrip('/') + '/api/' + path
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.request(
                method=request.method,
                url=target,
                headers={k: v for k, v in request.headers.items() if k.lower() not in ['host', 'connection']},
                params=request.query_params,
                content=await request.body(),
                cookies=request.cookies,
                follow_redirects=False,
                timeout=30,
            )
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=str(e))

    excluded_headers = ['content-encoding', 'content-length', 'transfer-encoding', 'connection']
    headers = {k: v for k, v in resp.headers.items() if k.lower() not in excluded_headers}
    if resp.headers.get('content-type', '').startswith('application/json'):
        try:
            return JSONResponse(content=resp.json(), status_code=resp.status_code, headers=headers)
        except Exception:
            # 如果 json 解析失敗，回傳文本
            return Response(content=resp.text, status_code=resp.status_code, headers=headers)
    else:
        # 回傳純文字或 HTML，使用透明的 Response 而非 JSONResponse
        return Response(content=resp.content, status_code=resp.status_code, headers=headers)


if __name__ == '__main__':
    import argparse
    import uvicorn
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', default=5500, type=int)
    args = parser.parse_args()
    print(f"Serving web UI on http://{args.host}:{args.port} — proxying API to {BACKEND}")
    uvicorn.run("web.web_server:app", host=args.host, port=args.port, reload=True)
