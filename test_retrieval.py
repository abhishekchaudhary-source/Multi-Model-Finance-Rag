import os
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http import models

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "Fintech Collection")


class FinancialMultiModalRetriever:
    def __init__(self):
        print(f"Connecting to Qdrant Cloud ({QDRANT_COLLECTION})...")
        self.embeddings = HuggingFaceBgeEmbeddings(
            model_name="BAAI/bge-small-en-v1.5",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
        self.client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        self.vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=QDRANT_COLLECTION,
            embedding=self.embeddings,
            vector_name="dense"
        )

    def retrieve(
        self,
        query: str,
        company: str = None,
        content_type: str = None,  # 'table', 'chart', 'text'
        year: str = None,
        quarter: str = None,
        doc_type: str = None,
        top_k: int = 3
    ):
        must_conditions = []
        if company:
            must_conditions.append(models.FieldCondition(key="metadata.company", match=models.MatchValue(value=company)))
        if content_type:
            must_conditions.append(models.FieldCondition(key="metadata.content_type", match=models.MatchValue(value=content_type)))
        if year:
            must_conditions.append(models.FieldCondition(key="metadata.year", match=models.MatchValue(value=str(year))))
        if quarter:
            must_conditions.append(models.FieldCondition(key="metadata.quarter", match=models.MatchValue(value=quarter.upper())))
        if doc_type:
            must_conditions.append(models.FieldCondition(key="metadata.doc_type", match=models.MatchValue(value=doc_type.upper())))

        q_filter = models.Filter(must=must_conditions) if must_conditions else None

        results = self.vector_store.similarity_search_with_score(query=query, k=top_k, filter=q_filter)
        return results


def display_results(query: str, results):
    print("\n" + "=" * 70)
    print(f"🔍 QUERY: {query}")
    print("=" * 70)

    if not results:
        print("  ⚠️ No matching documents found.")
        return

    for idx, (doc, score) in enumerate(results, 1):
        m = doc.metadata
        c_type = m.get("content_type", "unknown").upper()
        print(f"\n[Result {idx}] Score: {score:.4f} | Modality: {c_type}")
        print(f"Company: {m.get('company')} | Doc: {m.get('doc_type')} {m.get('year')} {m.get('quarter')} | Page: {m.get('page')}")
        if m.get("image_path"):
            print(f"🖼️ Image File: {m.get('image_path')} ({m.get('image_dimensions', '')})")
        print("-" * 50)
        print(doc.page_content[:400] + ("..." if len(doc.page_content) > 400 else ""))


if __name__ == "__main__":
    retriever = FinancialMultiModalRetriever()

    # Test 1: Financial Table query
    q1 = "Total net sales by product and services category"
    res1 = retriever.retrieve(query=q1, company="Apple", content_type="table", top_k=2)
    display_results(q1, res1)

    # Test 2: Chart / Graph query
    q2 = "Company stock performance comparison graph"
    res2 = retriever.retrieve(query=q2, company="Apple", content_type="chart", top_k=1)
    display_results(q2, res2)

    # Test 3: Narrative text query
    q3 = "What are the primary business risks and competition factors?"
    res3 = retriever.retrieve(query=q3, company="Amazon", content_type="text", top_k=2)
    display_results(q3, res3)
