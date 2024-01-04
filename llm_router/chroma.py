from typing import Optional

import chromadb

from llm_router import Engine
from chromadb.utils import embedding_functions


class ChromaDbEngine(Engine):
    def __init__(self, embedding_function, threshold, cache_path=None):
        """
        :param model_name: sentence_transformer model to use
        :param threshold: minimum similarity score to match a route
        """
        self.client = chromadb.PersistentClient(cache_path) if cache_path else chromadb.EphemeralClient()
        self.transformer_fn = embedding_function  # embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_name)
        self.threshold = threshold
        self.collection = self.client.get_or_create_collection(
            name="sentences",
            embedding_function=self.transformer_fn)

    def encode(self, routes):
        docs = []
        meta = []

        for r in routes:
            for s in r.sentences:
                docs.append(s)
                meta.append({"name": r.name})

        self.collection.upsert(
            documents=docs,
            metadatas=meta,
            ids=docs
        )

    def match(self, query):
        res = self.collection.query(
            query_texts=[query],
            n_results=3,
        )

        if len(res) == 0 or res['distances'][0][0] < self.threshold:
            return None

        return res['metadatas'][0][0]['name']


class SentenceTransformer(ChromaDbEngine):
    def __init__(self,
                 threshold=0,
                 cache_path=None,
                 model_name: str = "all-distilroberta-v1"):
        super().__init__(
            cache_path=cache_path,
            threshold=threshold,
            embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=model_name
            ))


class OpenAI(ChromaDbEngine):
    def __init__(self,
                 cache_path=None,
                 threshold=0,
                 api_key: Optional[str] = None,
                 model_name: str = "text-embedding-ada-002",
                 organization_id: Optional[str] = None,
                 api_base: Optional[str] = None,
                 api_type: Optional[str] = None,
                 api_version: Optional[str] = None,
                 deployment_id: Optional[str] = None):
        super().__init__(
            cache_path=cache_path,
            embedding_function=embedding_functions.OpenAIEmbeddingFunction(
                api_key=api_key,
                model_name=model_name,
                organization_id=organization_id,
                api_base=api_base,
                api_type=api_type,
                api_version=api_version,
                deployment_id=deployment_id
            ),
            threshold=threshold)
