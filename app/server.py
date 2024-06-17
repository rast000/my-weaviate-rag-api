from fastapi import BackgroundTasks, Body, Depends, HTTPException, UploadFile, FastAPI
from fastapi.responses import RedirectResponse
from langserve import add_routes
import os
import shutil
from rag_weaviate import chain as rag_weaviate_chain
import data.db as db

FILE_STORE = os.getenv("FILE_STORE", "files")

app = FastAPI()

def process_document(filename: str, file_path: str):
    db.add_file(filename, file_path)

@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/docs")

def is_pdf(doc: UploadFile):
    if not os.path.splitext(doc.filename)[-1] == ".pdf":
        raise HTTPException(status_code=400, detail="Invalid file type")

@app.post("/uploadfile", dependencies=[Depends(is_pdf)])
async def upload(doc: UploadFile, background_tasks: BackgroundTasks):
    os.makedirs(FILE_STORE, exist_ok=True)

    file_path = os.path.join(FILE_STORE, doc.filename)
    with open(file_path, "wb") as fp:
        shutil.copyfileobj(doc.file, fp)
    
    background_tasks.add_task(process_document, doc.filename, file_path)

    return {"filename": doc.filename}


# Edit this to add the chain you want to add
add_routes(app, rag_weaviate_chain, path="/rag-weaviate")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
