import os
import re
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from sentence_transformers import CrossEncoder
from qdrant_client import QdrantClient
from qdrant_client.http import models

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "Fintech Collection")


class FinancialMultiModalRerankRetriever:
    """
    Two-Stage Production Retrieval Pipeline:
    - Stage 1: Hybrid Search in Qdrant (Dense BGE-small + Keyword MatchText + RRF) fetches top candidates.
    - Stage 2: Neural Cross-Encoder Re-Ranking using BAAI/bge-reranker-v2-m3 (8192 context)
               evaluates deep query-document cross-attention and ranks top final results.
    """

    COMPANIES = ["Amazon", "Apple", "Google", "Meta"]

    def __init__(self, rrf_k: int = 60, reranker_model_name: Optional[str] = None):
        self.rrf_k = rrf_k
        embedding_model = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
        reranker_model = reranker_model_name or os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")

        hf_token = os.getenv("HF_TOKEN") or os.getenv("EMBEDDING_API_KEY")
        if hf_token:
            os.environ["HF_TOKEN"] = hf_token
            os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token

        print("=" * 65)
        print("🚀 INITIALIZING FINANCIAL MULTI-MODAL RETRIEVER & RE-RANKER")
        print("=" * 65)

        # 1. Stage 1 Bi-Encoder Embeddings
        print(f"1️⃣ Loading Bi-Encoder Embedding Model: {embedding_model}...")
        self.embeddings = HuggingFaceBgeEmbeddings(
            model_name=embedding_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
            query_instruction="Represent this sentence for searching relevant passages: "
        )

        # 2. Stage 2 Cross-Encoder Re-Ranker
        print(f"2️⃣ Loading Neural Cross-Encoder Re-Ranker: {reranker_model}...")
        self.reranker = CrossEncoder(reranker_model, max_length=1024)

        # 3. Connect to Qdrant Cloud
        print(f"3️⃣ Connecting to Qdrant Cloud Collection: '{QDRANT_COLLECTION}'...")
        self.client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        self._ensure_text_index()

        print("✅ Pipeline ready: Hybrid Search + BGE-Reranker-v2-m3 active!\n")

    def _ensure_text_index(self):
        """Ensures that page_content has a text payload index for fast keyword matching."""
        try:
            self.client.create_payload_index(
                collection_name=QDRANT_COLLECTION,
                field_name="page_content",
                field_schema=models.TextIndexParams(
                    type=models.TextIndexType.TEXT,
                    tokenizer=models.TokenizerType.WORD,
                    min_token_len=2,
                    max_token_len=25,
                    lowercase=True
                )
            )
        except Exception:
            pass

    UNINDEXED_COMPANIES = [
        "Tesla", "Uber", "Microsoft", "Netflix", "Nvidia", "Airbnb", 
        "Reliance", "Intel", "Salesforce", "Twitter", "Boeing", "IBM", 
        "Spotify", "Oracle", "Walmart", "Disney", "OpenAI"
    ]

    def parse_query_intent(self, user_query: str) -> Dict[str, Any]:
        """Extracts company, year, quarter, modality, and classified intent from user query."""
        from intent_classifier import FinancialIntentClassifier
        return FinancialIntentClassifier.classify(user_query)

    def retrieve_and_rerank(
        self,
        query: str,
        company: Optional[str] = None,
        content_type: Optional[str] = None,
        year: Optional[str] = None,
        quarter: Optional[str] = None,
        doc_type: Optional[str] = None,
        initial_candidates: Optional[int] = None,
        top_k: Optional[int] = None,
        dense_top_k: Optional[int] = None,
        sparse_top_k: Optional[int] = None,
        hybrid_top_k: Optional[int] = None,
        min_rerank_score: float = 0.25,
        apply_auto_filter: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Executes two-stage retrieval:
        1. Hybrid Search (Dense + Keyword/Sparse + RRF) retrieves candidate pool.
        2. CrossEncoder (`BAAI/bge-reranker-v2-m3`) re-scores each candidate with the query.
        3. Returns `top_k` highest-scoring chunks.
        """
        dense_limit = dense_top_k or int(os.getenv("DENSE_TOP_K", 12))
        sparse_limit = sparse_top_k or int(os.getenv("SPARSE_TOP_K", 12))
        hybrid_limit = hybrid_top_k or initial_candidates or int(os.getenv("HYBRID_TOP_K", 12))
        final_top_k = top_k if top_k is not None else int(os.getenv("RERANK_TOP_K", 3))

        detected = None
        if apply_auto_filter:
            detected = self.parse_query_intent(query)
            # Fast exit for greetings or unindexed companies
            if detected.get("intent") == "GREETING_CHITCHAT":
                return []
            if detected.get("unindexed_company") and not company:
                return []

            company = company or detected.get("company")
            year = year or detected.get("year")
            quarter = quarter or detected.get("quarter")
            content_type = content_type or detected.get("content_type")

        # --- STAGE 1: HYBRID RETRIEVAL ---
        must_conditions = []
        companies = detected.get("companies") if detected else []
        if len(companies) > 1 and not company:
            must_conditions.append(models.FieldCondition(key="metadata.company", match=models.MatchAny(any=companies)))
        elif company:
            must_conditions.append(models.FieldCondition(key="metadata.company", match=models.MatchValue(value=company)))

        if content_type:
            must_conditions.append(models.FieldCondition(key="metadata.content_type", match=models.MatchValue(value=content_type)))
        if year:
            must_conditions.append(models.FieldCondition(key="metadata.year", match=models.MatchValue(value=str(year))))
        if quarter:
            must_conditions.append(models.FieldCondition(key="metadata.quarter", match=models.MatchValue(value=quarter.upper())))
        if doc_type:
            must_conditions.append(models.FieldCondition(key="metadata.doc_type", match=models.MatchValue(value=doc_type.upper())))

        base_filter = models.Filter(must=must_conditions) if must_conditions else None

        # 1A. Dense Vector Search
        dense_vec = self.embeddings.embed_query(query)
        dense_hits = self.client.query_points(
            collection_name=QDRANT_COLLECTION,
            query=dense_vec,
            using="dense",
            query_filter=base_filter,
            limit=dense_limit
        ).points

        # 1B. Sparse / Keyword Search
        clean_words = [w for w in re.findall(r'\b\w+\b', query) if len(w) > 2 and w.lower() not in ["the", "and", "for", "what", "was", "show", "how", "much"]]
        kw_text = " ".join(clean_words[:5]) if clean_words else query

        kw_must = list(must_conditions)
        kw_must.append(models.FieldCondition(key="page_content", match=models.MatchText(text=kw_text)))
        kw_filter = models.Filter(must=kw_must)

        keyword_hits, _ = self.client.scroll(
            collection_name=QDRANT_COLLECTION,
            scroll_filter=kw_filter,
            limit=sparse_limit,
            with_payload=True
        )

        # 1C. RRF Fusion to collect diverse candidates
        candidates_map = {}
        for rank, hit in enumerate(dense_hits):
            p_id = hit.id
            candidates_map[p_id] = {
                "point": hit,
                "rrf_score": 1.0 / (self.rrf_k + rank + 1),
                "dense_rank": rank + 1,
                "kw_rank": None
            }

        for rank, hit in enumerate(keyword_hits):
            p_id = hit.id
            if p_id not in candidates_map:
                candidates_map[p_id] = {
                    "point": hit,
                    "rrf_score": 0.0,
                    "dense_rank": None,
                    "kw_rank": rank + 1
                }
            candidates_map[p_id]["rrf_score"] += 1.0 / (self.rrf_k + rank + 1)
            candidates_map[p_id]["kw_rank"] = rank + 1

        candidate_list = list(candidates_map.values())
        if not candidate_list:
            return []

        # Sort initial candidate pool by RRF to keep top pool
        candidate_list = sorted(candidate_list, key=lambda x: x["rrf_score"], reverse=True)[:hybrid_limit]

        # --- STAGE 2: NEURAL CROSS-ENCODER RE-RANKING ---
        pairs = []
        for c in candidate_list:
            doc_content = c["point"].payload.get("page_content", "")
            pairs.append((query, doc_content))

        # Compute deep cross-attention scores with BGE-Reranker-v2-m3
        rerank_scores = self.reranker.predict(pairs)

        # Attach reranker scores
        reranked_results = []
        for c, score in zip(candidate_list, rerank_scores):
            norm_score = float(score)
            if norm_score < min_rerank_score:
                continue

            p = c["point"]
            meta = p.payload.get("metadata", {})
            reranked_results.append({
                "rerank_score": round(norm_score, 5),
                "rrf_score": round(c["rrf_score"], 5),
                "dense_rank": c["dense_rank"],
                "keyword_rank": c["kw_rank"],
                "modality": meta.get("content_type", "text").upper(),
                "company": meta.get("company"),
                "filename": meta.get("filename"),
                "year": meta.get("year"),
                "quarter": meta.get("quarter"),
                "page": meta.get("page"),
                "image_path": meta.get("image_path"),
                "image_dimensions": meta.get("image_dimensions"),
                "content": p.payload.get("page_content")
            })

        # Sort by neural rerank score
        reranked_results = sorted(reranked_results, key=lambda x: x["rerank_score"], reverse=True)
        return reranked_results[:final_top_k]


def run_interactive_pipeline():
    retriever = FinancialMultiModalRerankRetriever()

    print("=" * 70)
    print("💬 Financial Multi-Modal RAG Search with BGE-Reranker-v2-m3")
    print("Stage 1: Qdrant Hybrid Search (Dense + Keyword)")
    print("Stage 2: BAAI/bge-reranker-v2-m3 Deep Cross-Attention Scoring")
    print("Type 'exit' to quit.")
    print("=" * 70 + "\n")

    while True:
        try:
            query = input("❓ Enter Query: ").strip()
            if not query:
                continue
            if query.lower() in ["exit", "quit"]:
                print("👋 Exiting pipeline. Goodbye!")
                break

            intent = retriever.parse_query_intent(query)
            print(f"🎯 Auto-detected Filters: {intent}")

            results = retriever.retrieve_and_rerank(query=query, top_k=3)

            if not results:
                print("⚠️ No relevant financial data found above threshold.\n")
                continue

            print(f"\n🏆 Top {len(results)} Re-Ranked Results (BAAI/bge-reranker-v2-m3):\n")
            for i, res in enumerate(results, 1):
                print(f"--- [Rank {i}] ⭐ Rerank Score: {res['rerank_score']} | Modality: {res['modality']} ---")
                print(f"Initial Stages: Dense Rank: {res['dense_rank']} | Keyword Rank: {res['keyword_rank']} | RRF: {res['rrf_score']}")
                print(f"Source: {res['company']} | {res['filename']} | Page: {res['page']}")
                if res['image_path']:
                    print(f"🖼️ Chart Image: {res['image_path']} ({res.get('image_dimensions', '')})")
                print("Content Preview:")
                print(res['content'][:350] + ("..." if len(res['content']) > 350 else ""))
                print("-" * 60 + "\n")

        except KeyboardInterrupt:
            print("\n👋 Terminated.")
            break
        except Exception as e:
            print(f"❌ Error: {e}\n")


if __name__ == "__main__":
    run_interactive_pipeline()
