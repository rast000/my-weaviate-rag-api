from .init import vectorstore, client
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter


def add_file(filename, file_path):
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    # pages = pages[:10] # limit number of pages
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(pages)

    uuid = None
    if not client.collections.exists("Documents"):
        client.collections.create("Documents")
    with client.batch.dynamic() as batch: # add file document
        uuid = batch.add_object(collection="Documents", properties={ "filename": filename })
    
    print("Document added", uuid)
    # print("Adding docs: ", docs)
    for doc in docs:
        doc.metadata["hasDocument"] = uuid
    vectorstore.add_documents(docs)

