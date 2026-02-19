import os
import sys
import json
import uuid

# Add parent directory to path so we can import sibling modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException, Request
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
COLLECTIONS_FILE = os.path.abspath(os.path.join(HERE, '..', 'collection_info.yaml'))  # path to collection.yaml

# === Agent (LLM, MCP) ===
global AGENT
global toolkit

LLM_Client = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key="XXX")
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


def load_collections():
    if os.path.exists(COLLECTIONS_FILE) and os.path.getsize(COLLECTIONS_FILE) > 0:
        if yaml:
            with open(COLLECTIONS_FILE, 'r', encoding='utf-8') as f:
                try:
                    data = yaml.safe_load(f) or {}
                except Exception:
                    data = {}
        else:
            # fallback to JSON if PyYAML missing
            with open(COLLECTIONS_FILE, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                except Exception:
                    data = {}
    else:
        data = {}
    # normalize structure
    if not isinstance(data, dict):
        data = {'collections': []}
    if 'collections' not in data:
        data['collections'] = []
    return data

def save_collections(data):
    # Ensure directory
    os.makedirs(os.path.dirname(COLLECTIONS_FILE), exist_ok=True)
    tmp = COLLECTIONS_FILE + '.tmp'
    if yaml:
        with open(tmp, 'w', encoding='utf-8') as f:
            yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
    else:
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, COLLECTIONS_FILE)

# === Startup Event ===
@app.on_event("startup")
async def startup_event():
    global toolkit, AGENT
    toolkit = await MCP_Client.get_tools()
    AGENT = Agent(LLM_Client, toolkit)
    print("✓ Agent initialized successfully")

@app.get('/api/collections')
def get_collections():
    data = load_collections()
    return data

@app.post('/api/collections')
async def create_collection(request: Request):
    payload = await request.json() or {}
    name = payload.get('name','').strip()
    description = payload.get('description','').strip()
    data = load_collections()
    cols = data.get('collections', [])
    if len(cols) >= 5:
        raise HTTPException(status_code=400, detail='Maximum of 5 collections reached')
    new_id = uuid.uuid4().hex
    col = {'id': new_id, 'name': name, 'description': description}
    cols.append(col)
    data['collections'] = cols
    save_collections(data)
    return JSONResponse(col, status_code=201)


@app.put('/api/collections/{col_id}')
async def update_collection(col_id: str, request: Request):
    data = load_collections()
    cols = data.get('collections', [])
    found = next((c for c in cols if c.get('id') == col_id), None)
    if not found:
        raise HTTPException(status_code=404, detail='Collection not found')
    payload = await request.json() or {}
    found['name'] = payload.get('name', found.get('name','')).strip()
    found['description'] = payload.get('description', found.get('description','')).strip()
    save_collections(data)
    return found


@app.delete('/api/collections/{col_id}')
def delete_collection(col_id: str):
    data = load_collections()
    cols = data.get('collections', [])
    found = next((c for c in cols if c.get('id') == col_id), None)
    if not found:
        raise HTTPException(status_code=404, detail='Collection not found')
    cols = [c for c in cols if c.get('id') != col_id]
    data['collections'] = cols
    save_collections(data)
    return JSONResponse({}, status_code=204)


@app.post('/api/chat')
async def agent_response(request: ChatRequest):
    query = request.query
    try:
        response = await AGENT.chat(query)
    except:
        raise HTTPException(status_code=505, detail='Agent has error during response genreation') 
    
    return response["output"]


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
    return JSONResponse(resp.json() if resp.headers.get('content-type', '').startswith('application/json') else resp.text, 
                       status_code=resp.status_code, headers=headers)


if __name__ == '__main__':
    import argparse
    import uvicorn
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', default=5500, type=int)
    args = parser.parse_args()
    print(f"Serving web UI on http://{args.host}:{args.port} — proxying API to {BACKEND}")
    uvicorn.run("web.web_server:app", host=args.host, port=args.port, reload=True)
