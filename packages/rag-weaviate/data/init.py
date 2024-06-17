import weaviate
from langchain_weaviate.vectorstores import WeaviateVectorStore
from langchain_community.embeddings.openai import OpenAIEmbeddings
import os

if os.environ.get("WEAVIATE_API_KEY", None) is None:
    raise Exception("Missing `WEAVIATE_API_KEY` environment variable.")

if os.environ.get("WEAVIATE_ENVIRONMENT", None) is None:
    raise Exception("Missing `WEAVIATE_ENVIRONMENT` environment variable.")

client: weaviate.WeaviateClient | None = None
vectorstore: WeaviateVectorStore | None = None


def init_db():
    """Connect to SQLite database file"""
    global client, vectorstore
    if client:
        return

    client = weaviate.connect_to_wcs(
        cluster_url=os.environ["WEAVIATE_API_URL"],
        auth_credentials=weaviate.AuthApiKey(api_key=os.environ["WEAVIATE_API_KEY"]),  # Replace w/ your Weaviate instance API key
        headers={"X-OpenAI-Api-Key": os.environ["OPENAI_API_KEY"]}
    )

    embeddings = OpenAIEmbeddings()
    vectorstore = WeaviateVectorStore(
        client, os.environ["WEAVIATE_INDEX"], "text", embedding=embeddings
    )


init_db()
