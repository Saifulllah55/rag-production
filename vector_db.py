import os

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

from dotenv import load_dotenv

load_dotenv()


class QdrantStorage:
    def __init__(
        self,
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
        collection="docs",
        dim=3072,
    ):
        if api_key is None:
            api_key = os.getenv("QDRANT_API_KEY")

        self.client = QdrantClient(url=url, api_key=api_key, timeout=30)
        self.collection = collection

        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(
                    size=dim,
                    distance=Distance.COSINE
                ),
            )

    def upsert(self, ids, vectors, payloads):
        points = [
            PointStruct(
                id=ids[i],
                vector=vectors[i],
                payload=payloads[i]
            )
            for i in range(len(ids))
        ]

        self.client.upsert(
            self.collection,
            points=points
        )

    def search(self, query_vector, top_k: int = 5):
        results = self.client.query_points(
            collection_name=self.collection,
            query=query_vector,
            with_payload=True,
            limit=top_k
        )

        contexts = []
        sources = set()

        for r in results.points:
            payload = r.payload or {}
            text = payload.get("text", "")
            source = payload.get("source", "")

            if text:
                contexts.append(text)
                sources.add(source)

        return {
            "contexts": contexts,
            "sources": list(sources)
        }