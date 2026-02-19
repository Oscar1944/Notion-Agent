# Notion-Agent Web UI

This folder contains a minimal single-page web UI for the Notion-Agent project.

Files:
- `index.html` — the single-page frontend.
- `styles.css` — styling for the UI.
- `app.js` — frontend logic (sends a POST to `/api/query`).
- `web_server.py` — optional Flask app that serves the static files and proxies `/api/*` to a backend.

Quick start (static only):

1. Open `index.html` in a browser (or serve with a static server):

```bash
cd "Notion-Agent/web"
python -m http.server 5500
# then open http://localhost:5500
```

Quick start (Flask proxy):

1. (optional) Create/activate a Python virtualenv and install dependencies:

```bash
pip install flask requests pyyaml
```

2. Run the proxy (set `BACKEND_URL` to your Notion-Agent backend if needed):

```bash
cd "Notion-Agent/web"
set BACKEND_URL=http://localhost:8000
python web_server.py --host 0.0.0.0 --port 5500
# open http://localhost:5500
```

Usage notes:
- The frontend POSTs JSON {"query":"..."} to `/api/query` on the backend. Adjust `app.js` if your API differs.
- The Flask proxy forwards `/api/*` to the URL in `BACKEND_URL` and serves the static files.

Collections API:
- `GET /api/collections` — returns `{ collections: [ {id,name,description} ] }`
- `POST /api/collections` — create new collection with JSON `{name,description}` (max 5)
- `PUT /api/collections/<id>` — update collection
- `DELETE /api/collections/<id>` — delete collection

The web UI will save collection metadata to `collection_info.yaml` in the `Notion-Agent` folder.

If you'd like, I can:
- Wire the UI to the exact endpoints in `tool/mcp_server.py`.
- Add authentication, file upload, or a result history panel.
