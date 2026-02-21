document.addEventListener('DOMContentLoaded',()=>{
  const backendInput=document.getElementById('backend');
  const queryEl=document.getElementById('query');
  const sendBtn=document.getElementById('send');
  const clearBtn=document.getElementById('clear');
  const respEl=document.getElementById('response');

  // Dossier (file upload) UI
  const fileInput = document.getElementById('file-input');
  const uploadBtn = document.getElementById('upload-file');
  const fileList = document.getElementById('file-list');

  function setResponse(text){ 
    if (!text) return;
    
    const respEl = document.getElementById('response');
    
    // Regular expression: replace char \\n with change line \n 
    let cleanText = String(text).replace(/\\n/g, '\n');
    cleanText = cleanText.replace(/\\n\\n/g, '\n\n');

    respEl.textContent = cleanText;
  }

  // Helper: build backend base
  function backendBase(){ return (backendInput.value||window.location.origin).replace(/\/+$/,'') }

  clearBtn.addEventListener('click',()=>{ queryEl.value=''; setResponse('No response yet.') })

  // Fetch dossier files
  async function fetchFiles(){
    try{
      const r = await fetch(backendBase() + '/api/dossier/files')
      if(!r.ok) throw new Error(await r.text())
      const json = await r.json()
      renderFiles(json.files || [])
    }catch(err){ fileList.innerHTML = '<li class="error">Error loading files: '+escapeHtml(err.message)+'</li>' }
  }

  function renderFiles(list){
    fileList.innerHTML = ''
    if(list.length === 0){ fileList.innerHTML = '<li class="muted">No files uploaded.</li>' }
    list.forEach(f=>{
      const li = document.createElement('li')
      li.className = 'file-item'
      const nameEsc = escapeHtml(f.name || f)
      const sizeInfo = f.size ? ` <small class="muted">(${f.size} bytes)</small>` : ''
      li.innerHTML = `<div class="meta"><a href="/uploads/${encodeURIComponent(f.name)}" target="_blank">${nameEsc}</a>${sizeInfo}</div><div class="ctrls"><button class="delete-file" data-name="${escapeHtml(f.name)}">Delete</button></div>`
      fileList.appendChild(li)
    })
    Array.from(fileList.querySelectorAll('button.delete-file')).forEach(b=>b.addEventListener('click', async e=>{
      const name = e.currentTarget.dataset.name
      if(!confirm('Delete this file?')) return
      await deleteFile(name)
    }))
  }

  function escapeHtml(s){ return String(s).replace(/[&<>\"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c])) }

  // Upload file
  async function uploadFile(){
    const file = fileInput.files && fileInput.files[0]
    if(!file){ setResponse('Select a file to upload'); return }
    const form = new FormData()
    form.append('file', file)
    try{
      const r = await fetch(backendBase() + '/api/dossier/files', {method:'POST', body: form})
      if(!r.ok) throw new Error(await r.text())
      setResponse('Upload successful')
      fileInput.value = ''
      await fetchFiles()
    }catch(err){ setResponse('Upload failed: '+err.message) }
  }

  // Delete file
  async function deleteFile(name){
    try{
      const r = await fetch(backendBase() + '/api/dossier/files/' + encodeURIComponent(name), {method:'DELETE'})
      if(!r.ok) throw new Error(await r.text())
      setResponse('File deleted')
      await fetchFiles()
    }catch(err){ setResponse('Delete failed: '+err.message) }
  }

  uploadBtn.addEventListener('click', uploadFile)

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
      const text=await agent_response.json()
      setResponse(text)
    }catch(err){ setResponse('Request failed: '+err.message) }
  })

  // Initial load
  fetchFiles()
})
