document.addEventListener('DOMContentLoaded',()=>{
  const backendInput=document.getElementById('backend');
  const queryEl=document.getElementById('query');
  const sendBtn=document.getElementById('send');
  const clearBtn=document.getElementById('clear');
  const respEl=document.getElementById('response');

  // Collections UI
  const collectionList=document.getElementById('collection-list');
  const addCollectionBtn=document.getElementById('add-collection');
  const collectionForm=document.getElementById('collection-form');
  const colName=document.getElementById('col-name');
  const colDesc=document.getElementById('col-desc');
  const saveCollectionBtn=document.getElementById('save-collection');
  const cancelCollectionBtn=document.getElementById('cancel-collection');
  let editingId = null;

  function setResponse(text){ respEl.textContent=text }

  function showForm(collection){
    collectionForm.classList.remove('hidden');
    if(collection){ editingId = collection.id; colName.value = collection.name || ''; colDesc.value = collection.description || ''; }
    else { editingId = null; colName.value = ''; colDesc.value = ''; }
  }
  function hideForm(){ collectionForm.classList.add('hidden'); editingId = null }

  addCollectionBtn.addEventListener('click', ()=> showForm(null));
  cancelCollectionBtn.addEventListener('click', ()=> hideForm());

  clearBtn.addEventListener('click',()=>{ queryEl.value=''; setResponse('No response yet.') })

  async function fetchCollections(){
    const backend = (backendInput.value||window.location.origin).replace(/\/+$/,'')
    try{
      const r = await fetch(backend + '/api/collections')
      if(!r.ok) throw new Error(await r.text())
      const json = await r.json()
      renderCollections(json.collections || [])
    }catch(err){ collectionList.innerHTML = '<li class="error">Error loading collections: '+escapeHtml(err.message)+'</li>' }
  }

  function renderCollections(list){
    collectionList.innerHTML = ''
    if(list.length === 0){ collectionList.innerHTML = '<li class="muted">No collections yet.</li>' }
    list.forEach(col=>{
      const li = document.createElement('li')
      li.className = 'collection-item'
      li.innerHTML = `<div class="meta"><strong>${escapeHtml(col.name||'Untitled')}</strong><div class="desc">${escapeHtml(col.description||'')}</div></div><div class="ctrls"><button class="edit" data-id="${col.id}">Edit</button><button class="delete" data-id="${col.id}">Delete</button></div>`
      collectionList.appendChild(li)
    })
    // disable add when >=5
    addCollectionBtn.disabled = list.length >= 5
    Array.from(collectionList.querySelectorAll('button.edit')).forEach(b=>b.addEventListener('click', async e=>{
      const id = e.currentTarget.dataset.id
      const col = list.find(x=>x.id===id)
      showForm(col)
    }))
    Array.from(collectionList.querySelectorAll('button.delete')).forEach(b=>b.addEventListener('click', async e=>{
      const id = e.currentTarget.dataset.id
      if(!confirm('Delete this collection?')) return
      await deleteCollection(id)
    }))
  }

  function escapeHtml(s){ return String(s).replace(/[&<>\"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c])) }

  async function saveCollection(){
    const backend = (backendInput.value||window.location.origin).replace(/\/+$/,'')
    const payload = { name: colName.value.trim(), description: colDesc.value.trim() }
    try{
      let r
      if(editingId){ r = await fetch(`${backend}/api/collections/${editingId}`, {method:'PUT',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)}) }
      else { r = await fetch(`${backend}/api/collections`, {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)}) }
      if(!r.ok) throw new Error(await r.text())
      setResponse('Collection saved')
      hideForm()
      await fetchCollections()
    }catch(err){ setResponse('Save failed: '+err.message) }
  }

  async function deleteCollection(id){
    const backend = (backendInput.value||window.location.origin).replace(/\/+$/,'')
    try{
      const r = await fetch(`${backend}/api/collections/${id}`, {method:'DELETE'})
      if(!r.ok) throw new Error(await r.text())
      setResponse('Collection deleted')
      await fetchCollections()
    }catch(err){ setResponse('Delete failed: '+err.message) }
  }

  saveCollectionBtn.addEventListener('click', saveCollection)

  // ==== Sending query to Agent ====
  sendBtn.addEventListener('click',async()=>{
    const backend=(backendInput.value||window.location.origin).replace(/\/+$/,'')
    const query=queryEl.value.trim()
    if(!query){ setResponse('Enter a query first'); return }

    setResponse('Sending...')
    try{
      // Default endpoint: POST { query: string } to /api/chat
      const url=backend+ '/api/chat'
      const agent_response=await fetch(url,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({query})})
      const text=await agent_response.text()
      try{ // pretty JSON when possible
        const json=JSON.parse(text)
        setResponse(JSON.stringify(json,null,2))
      }catch(e){ setResponse(text) }
    }catch(err){ setResponse('Request failed: '+err.message) }
  })

  // Initial load
  fetchCollections()
})
