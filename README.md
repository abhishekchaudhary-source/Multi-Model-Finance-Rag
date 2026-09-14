# 📈 Multi-Modal Enterprise Financial RAG System

An enterprise-grade, multi-modal Retrieval-Augmented Generation (RAG) system specialized in regulatory financial filings (SEC Form 10-K, 10-Q, 8-K) for major tech corporations: **Apple, Amazon, Google, and Meta**.

---

## 🚀 Key Architectural Capabilities

1. **Two-Stage Hybrid Retrieval & Neural Re-ranking:**
   - **Stage 1 (Dense + Keyword Search):** Bi-Encoder (`BAAI/bge-small-en-v1.5`) dense vector search combined with BM25-style keyword search over Qdrant Cloud.
   - **Stage 2 (Neural Cross-Encoder):** Neural re-ranking via `BAAI/bge-reranker-v2-m3` evaluates deep query-document cross-attention to select the most relevant chunks.
2. **Multi-Modal Generation Layer:**
   - Powered by **Google Gemini (Flash)** for grounded reasoning across textual narrative, financial markdown tables, and extracted chart/graph images.
3. **Deterministic Financial Guardrails:**
   - **Security Guardrail:** Blocks prompt injections, malicious jailbreaks, and out-of-domain queries.
   - **Groundedness & Fact-Checking:** Validates that numerical entities, dollar amounts, and percentages in answers strictly match filing citations.
   - **Multi-Tier Fallbacks:** Clean handling for low confidence, out-of-scope queries, or offline API scenarios.
4. **Relational Persistence & Telemetry (PostgreSQL 18):**
   - Tracks session chats, request latency, estimated query cost, and 5 evaluation metrics:
     - **Faithfulness Score**
     - **Context Relevance**
     - **Retrieval Recall**
     - **Response Coherence**
     - **Citation Accuracy**
5. **Modern Gemini-Themed Streamlit UI:**
   - Clean financial analysis interface with real-time scorecard metrics, auto-generated sample queries, source verification, and PostgreSQL audit indicators.

---

## 🛠️ Tech Stack

- **Vector Database:** [Qdrant Cloud](https://qdrant.tech/)
- **Relational Database:** PostgreSQL 18 with SQLAlchemy pooling
- **LLM Synthesis:** Google Gemini (`gemini-3.6-flash` / `gemini-2.0-flash`)
- **Embeddings:** `BAAI/bge-small-en-v1.5`
- **Re-Ranker:** `BAAI/bge-reranker-v2-m3`
- **Frontend / Chat:** Streamlit
- **Framework:** LangChain & Python 3.14

---

## ⚙️ Quickstart & Setup

### 1. Clone the repository
```bash
git clone https://github.com/abhishek-795/Multi-Model-Finance-Rag.git
cd Multi-Model-Finance-Rag
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables
Copy `.env.example` to `.env` and fill in your credentials:
```bash
cp .env.example .env
```

```env
# Vector Database
QDRANT_URL="https://your-cluster-url.qdrant.io"
QDRANT_API_KEY="your-qdrant-key"
QDRANT_COLLECTION="Fintech Collection"

# Relational Database
DATABASE_URL="postgresql://postgres:password@localhost:5432/finance_rag"

# Models & Keys
EMBEDDING_MODEL="BAAI/bge-small-en-v1.5"
RERANKER_MODEL="BAAI/bge-reranker-v2-m3"
LLM_MODEL="gemini-3.6-flash"
GEMINI_API_KEY="your-gemini-api-key"
```

### 4. Run the Application
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser.
