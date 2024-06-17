import os
from typing import Any, Dict, Iterable, List, Optional, Type

from langchain_community.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.runnables import (
    ConfigurableField,
    RunnableConfig,
    RunnableSerializable,
    RunnableParallel,
    RunnablePassthrough,
)
from langchain_core.runnables.utils import Input
from langchain_weaviate import WeaviateVectorStore
from weaviate.classes.query import Filter
from langchain_core.documents import Document
from data.init import vectorstore

# RAG prompt
template = """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
Question: {question}
Context: {context}
Answer:
"""
prompt = ChatPromptTemplate.from_template(template)


class FilteredRetriever(RunnableSerializable):
    document_id: Optional[str]
    vectorstore: WeaviateVectorStore

    class Config:
            arbitrary_types_allowed = True

    def _invoke(
        self, input: str, config: Optional[RunnableConfig] = None, **kwargs: Any
    ) -> List[Document]:
        # document_object = self.vectorstore._client.collections.get("Documents").query.fetch_object_by_id(self.document_id)
        retriever = self.vectorstore.as_retriever(
            search_kwargs={
                "filters": Filter.by_property("hasDocument").equal(self.document_id)
            }
        )
        return retriever.invoke(input, config=config)

    def invoke(
        self, input: str, config: Optional[RunnableConfig] = None, **kwargs
    ) -> List[Document]:
        return self._call_with_config(self._invoke, input, config, **kwargs)


# RAG
model = ChatOpenAI()
chain = (
    RunnableParallel(
        {
            "context": FilteredRetriever(
                vectorstore=vectorstore, document_id=None
            ).configurable_fields(
                document_id=ConfigurableField(
                    id="document_id",
                    name="Document",
                    description="Document for context",
                )
            ),
            "question": RunnablePassthrough(),
        }
    )
    | prompt
    | model
    | StrOutputParser()
)


# Add typing for input
class Question(BaseModel):
    __root__: str


chain = chain.with_types(input_type=Question)
