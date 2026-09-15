import os
import sys
import time
import re
from typing import Optional, List, Dict, Any
from pathlib import Path

# Ensure project root in sys.path
PROJECT_ROOT = str(Path(__file__).resolve().parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Core project modules
from intent_classifier import FinancialIntentClassifier, get_conversational_greeting
from database import (
    is_db_connected,
    log_query_audit,
    save_chat_message,
    get_recent_audit_logs,
    get_analytics_summary
)
from metrics import FinancialRAGMetrics

app = FastAPI(
    title="Multi-Model Financial RAG API & Test Console",
    description="Enterprise API and interactive testing suite for SEC Financial Intelligence with Intent Routing and Real-time Auditing.",
    version="2.0.0"
)

# CORS middleware for open testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Data directory for visual chart assets
if os.path.exists(os.path.join(PROJECT_ROOT, "Data")):
    app.mount("/data", StaticFiles(directory=os.path.join(PROJECT_ROOT, "Data")), name="data")

# Initialize lightweight guardrails and fallback manager
from rag_engine import FinancialGuardrails, FallbackManager
_guardrails = FinancialGuardrails()
_fallback = FallbackManager()

_retriever = None
_generator = None

def get_retriever_and_generator():
    global _retriever, _generator
    if _retriever is None:
        from retriever import FinancialMultiModalRerankRetriever
        from generator import GeminiFinancialGenerator
        _retriever = FinancialMultiModalRerankRetriever()
        _generator = GeminiFinancialGenerator()
    return _retriever, _generator


# --- Pydantic Schemas ---
class ClassifyRequest(BaseModel):
    query: str = Field(..., example="What was Apple's iPhone net sales in Q2 2024?")

class QueryRequest(BaseModel):
    query: str = Field(..., example="What was Apple's iPhone net sales in Q2 2024?")
    top_k: Optional[int] = Field(default=3, ge=1, le=10)
    provider: Optional[str] = Field(default="gemini", description="LLM provider: 'gemini' or 'mistral'")
    session_id: Optional[str] = Field(default="fastapi-tester")


# --- REST API Endpoints ---
@app.get("/api/health", tags=["Health"])
def health_check():
    """Returns real-time health status of PostgreSQL, Qdrant Cloud, and Gemini."""
    db_ok = is_db_connected()
    return {
        "status": "online",
        "service": "Multi-Model Financial RAG API",
        "components": {
            "postgresql": "connected" if db_ok else "offline",
            "qdrant_cloud": "configured" if os.getenv("QDRANT_URL") else "missing",
            "gemini_llm": "configured" if os.getenv("GEMINI_API_KEY") else "missing",
            "embedding_model": os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"),
            "reranker_model": os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
        },
        "timestamp": time.time()
    }


@app.post("/api/classify", tags=["Intent Classifier"])
def classify_intent(req: ClassifyRequest):
    """
    Classifies a user query into one of 5 Financial Intent categories:
    - GREETING_CHITCHAT
    - OUT_OF_SCOPE
    - VISUAL_CHART_REQUEST
    - CROSS_COMPANY_COMPARISON
    - FINANCIAL_NUMERICAL_ANALYSIS
    """
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    result = FinancialIntentClassifier.classify(req.query.strip())
    return {
        "query": req.query,
        "classification": result
    }


@app.post("/api/query", tags=["RAG Execution"])
def execute_query(req: QueryRequest):
    """
    Executes end-to-end RAG with Intent Classification, Guardrails,
    Hybrid Retrieval, Neural Re-Ranking, Gemini Generation, and Telemetry Logging.
    """
    cleaned_query = req.query.strip()
    if not cleaned_query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    start_time = time.time()

    # 1. Guardrail Check
    input_check = _guardrails.validate_input(cleaned_query)
    if not input_check["passed"]:
        sec_msg = f"🛡️ Security Guardrail Triggered: {input_check['reason']}"
        log_query_audit(
            query=cleaned_query,
            answer=sec_msg,
            metrics=None,
            sources=[],
            session_id=req.session_id,
            is_fallback=True,
            intent="GUARDRAIL_BLOCKED"
        )
        return {
            "query": cleaned_query,
            "answer": sec_msg,
            "intent": {
                "intent": "GUARDRAIL_BLOCKED",
                "badge_label": "🛡️ Guardrail Blocked",
                "badge_color": "#EA4335",
                "confidence": 1.0
            },
            "sources": [],
            "metrics": None,
            "latency_seconds": round(time.time() - start_time, 3)
        }

    # 2. Intent Classification
    intent_data = FinancialIntentClassifier.classify(cleaned_query)
    intent_type = intent_data["intent"]

    # 3. Intent-Driven Routing
    if intent_type == "GREETING_CHITCHAT":
        answer_text = get_conversational_greeting()
        chunks = []
        chart_imgs = []
    elif intent_type == "OUT_OF_SCOPE":
        answer_text = _fallback.out_of_scope_fallback(cleaned_query)
        chunks = []
        chart_imgs = []
    else:
        retriever, generator = get_retriever_and_generator()
        # Visual or Numerical or Cross-Company Retrieval
        retrieve_kwargs = {
            "query": cleaned_query,
            "top_k": req.top_k,
            "dense_top_k": int(os.getenv("DENSE_TOP_K", 12)),
            "sparse_top_k": int(os.getenv("SPARSE_TOP_K", 12)),
            "hybrid_top_k": int(os.getenv("HYBRID_TOP_K", 12)),
            "min_rerank_score": 0.20
        }
        if intent_type == "VISUAL_CHART_REQUEST":
            retrieve_kwargs["content_type"] = "chart"

        chunks = retriever.retrieve_and_rerank(**retrieve_kwargs)

        if not chunks:
            answer_text = fallback.out_of_scope_fallback(cleaned_query)
            chart_imgs = []
        else:
            gen_result = generator.generate(cleaned_query, chunks, provider=req.provider)
            answer_text = gen_result["answer"]
            chart_imgs = []
            for c in chunks:
                if c.get("image_path") and os.path.exists(c["image_path"]):
                    rel_path = c["image_path"].replace(PROJECT_ROOT + "/", "")
                    if not rel_path.startswith("/"):
                        rel_path = "/" + rel_path
                    chart_imgs.append({
                        "url": rel_path.replace("/Data/", "/data/"),
                        "title": f"{c['company']} {c.get('year', '')} Visual Evidence",
                        "filename": c.get("filename", ""),
                        "page": c.get("page", ""),
                        "score": c.get("rerank_score", "")
                    })

    # 4. Telemetry Metrics Evaluation
    metrics_eval = FinancialRAGMetrics.evaluate_all(
        query=cleaned_query,
        answer=answer_text,
        context_chunks=chunks,
        start_time=start_time
    )

    is_fb = metrics_eval.get("is_fallback") or (intent_type == "OUT_OF_SCOPE")

    # 5. Persist to PostgreSQL
    save_chat_message(
        role="assistant",
        content=answer_text,
        session_id=req.session_id,
        metadata={"intent": intent_type, "num_sources": len(chunks)}
    )
    audit_id = log_query_audit(
        query=cleaned_query,
        answer=answer_text,
        metrics=metrics_eval,
        sources=chunks,
        session_id=req.session_id,
        is_fallback=is_fb,
        intent=intent_type
    )

    return {
        "query": cleaned_query,
        "answer": answer_text,
        "intent": intent_data,
        "sources": [
            {
                "company": c.get("company"),
                "filename": c.get("filename"),
                "page": c.get("page"),
                "modality": c.get("modality"),
                "rerank_score": c.get("rerank_score"),
                "content_preview": c.get("content", "")[:280] + "..."
            }
            for c in chunks
        ],
        "charts": chart_imgs,
        "metrics": metrics_eval,
        "audit_id": audit_id,
        "latency_seconds": round(time.time() - start_time, 3)
    }


@app.get("/api/audits", tags=["Audits"])
def list_recent_audits(limit: int = 10):
    """Returns latest query audit entries logged to PostgreSQL."""
    return get_recent_audit_logs(limit=limit)


@app.get("/api/analytics", tags=["Audits"])
def get_analytics():
    """Returns aggregate analytics metrics from PostgreSQL."""
    return get_analytics_summary()


# --- INTERACTIVE TESTING FRONTEND ---
@app.get("/", response_class=HTMLResponse, tags=["Frontend"])
def serve_testing_dashboard():
    """Serves the rich interactive testing UI for FastAPI."""
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta name="description" content="Multi-Model Financial RAG Enterprise Interactive Testing Console and API Playground">
    <title>Financial RAG | API Testing Console</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-body: #0d1117;
            --bg-surface: #161b22;
            --bg-card: #21262d;
            --border: #30363d;
            --border-glow: #388bfd40;
            --text: #f0f6fc;
            --text-muted: #8b949e;
            --primary: #58a6ff;
            --primary-hover: #79c0ff;
            --success: #3fb950;
            --warning: #d29922;
            --danger: #f85149;
            --purple: #bc8cff;
            --font-sans: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            --font-mono: 'JetBrains Mono', monospace;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            background-color: var(--bg-body);
            color: var(--text);
            font-family: var(--font-sans);
            line-height: 1.5;
            -webkit-font-smoothing: antialiased;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
        }

        /* Top Navbar */
        .navbar {
            background-color: var(--bg-surface);
            border-bottom: 1px solid var(--border);
            padding: 14px 28px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            position: sticky;
            top: 0;
            z-index: 50;
        }

        .brand {
            display: flex;
            align-items: center;
            gap: 12px;
        }

        .brand-icon {
            width: 34px;
            height: 34px;
            background: linear-gradient(135deg, #1f6feb 0%, #a371f7 100%);
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 18px;
            font-weight: 700;
            box-shadow: 0 0 14px rgba(31, 111, 235, 0.4);
        }

        .brand-title {
            font-size: 1.15rem;
            font-weight: 700;
            letter-spacing: -0.02em;
        }

        .brand-badge {
            background: rgba(88, 166, 255, 0.15);
            color: var(--primary);
            font-size: 0.75rem;
            font-weight: 600;
            padding: 2px 8px;
            border-radius: 12px;
            border: 1px solid rgba(88, 166, 255, 0.3);
        }

        .nav-status {
            display: flex;
            align-items: center;
            gap: 16px;
            font-size: 0.85rem;
        }

        .status-pill {
            display: flex;
            align-items: center;
            gap: 6px;
            background: var(--bg-card);
            border: 1px solid var(--border);
            padding: 4px 12px;
            border-radius: 20px;
            color: var(--text-muted);
        }

        .status-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: var(--success);
            box-shadow: 0 0 8px var(--success);
        }

        .nav-btn {
            background: var(--primary);
            color: #0d1117;
            text-decoration: none;
            font-weight: 600;
            padding: 6px 14px;
            border-radius: 6px;
            font-size: 0.85rem;
            transition: all 0.2s ease;
        }

        .nav-btn:hover {
            background: var(--primary-hover);
            box-shadow: 0 0 12px rgba(88, 166, 255, 0.4);
        }

        /* Container Layout */
        .main-container {
            flex: 1;
            max-width: 1300px;
            width: 100%;
            margin: 0 auto;
            padding: 24px;
            display: flex;
            flex-direction: column;
            gap: 20px;
        }

        /* Tabs */
        .tabs-header {
            display: flex;
            gap: 8px;
            border-bottom: 1px solid var(--border);
            padding-bottom: 8px;
        }

        .tab-btn {
            background: transparent;
            border: none;
            color: var(--text-muted);
            padding: 8px 16px;
            font-size: 0.95rem;
            font-weight: 600;
            cursor: pointer;
            border-radius: 6px;
            transition: all 0.2s;
            font-family: inherit;
        }

        .tab-btn.active {
            background: var(--bg-surface);
            color: var(--primary);
            border: 1px solid var(--border);
        }

        .tab-btn:hover:not(.active) {
            color: var(--text);
            background: rgba(255,255,255,0.03);
        }

        .tab-content {
            display: none;
        }

        .tab-content.active {
            display: block;
        }

        /* Test Console Grid */
        .console-grid {
            display: grid;
            grid-template-columns: 1.15fr 0.85fr;
            gap: 24px;
        }

        @media (max-width: 992px) {
            .console-grid {
                grid-template-columns: 1fr;
            }
        }

        .panel {
            background: var(--bg-surface);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 20px;
            display: flex;
            flex-direction: column;
            gap: 16px;
        }

        .panel-title {
            font-size: 1.05rem;
            font-weight: 600;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }

        /* Quick Prompt Buttons */
        .quick-prompts {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
        }

        .chip-btn {
            background: var(--bg-card);
            border: 1px solid var(--border);
            color: var(--text);
            font-size: 0.78rem;
            padding: 5px 11px;
            border-radius: 16px;
            cursor: pointer;
            transition: all 0.15s;
            font-family: inherit;
        }

        .chip-btn:hover {
            border-color: var(--primary);
            color: var(--primary);
            transform: translateY(-1px);
        }

        /* Input Controls */
        .query-input-box {
            display: flex;
            flex-direction: column;
            gap: 10px;
        }

        textarea {
            width: 100%;
            background: var(--bg-body);
            border: 1px solid var(--border);
            color: var(--text);
            padding: 12px 14px;
            border-radius: 8px;
            font-family: inherit;
            font-size: 0.95rem;
            resize: vertical;
            min-height: 80px;
            outline: none;
            transition: border-color 0.2s, box-shadow 0.2s;
        }

        textarea:focus {
            border-color: var(--primary);
            box-shadow: 0 0 0 3px var(--border-glow);
        }

        .action-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .btn-primary {
            background: var(--primary);
            color: #0d1117;
            border: none;
            padding: 9px 20px;
            font-size: 0.92rem;
            font-weight: 600;
            border-radius: 6px;
            cursor: pointer;
            display: flex;
            align-items: center;
            gap: 8px;
            transition: all 0.2s;
            font-family: inherit;
        }

        .btn-primary:hover {
            background: var(--primary-hover);
            box-shadow: 0 0 14px rgba(88, 166, 255, 0.4);
        }

        .btn-secondary {
            background: var(--bg-card);
            color: var(--text);
            border: 1px solid var(--border);
            padding: 8px 16px;
            font-size: 0.88rem;
            font-weight: 500;
            border-radius: 6px;
            cursor: pointer;
            transition: all 0.2s;
            font-family: inherit;
        }

        .btn-secondary:hover {
            background: #28303a;
        }

        /* Intent Badge Display */
        .intent-badge-card {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 12px 16px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            transition: all 0.3s ease;
        }

        .badge-pill {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            font-size: 0.82rem;
            font-weight: 600;
            padding: 4px 12px;
            border-radius: 14px;
            border: 1px solid transparent;
        }

        /* Answer Output Area */
        .answer-box {
            background: var(--bg-body);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 16px;
            font-size: 0.95rem;
            white-space: pre-wrap;
            min-height: 120px;
            max-height: 420px;
            overflow-y: auto;
            position: relative;
        }

        /* Metrics Card */
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 12px;
        }

        .metric-cell {
            background: var(--bg-body);
            border: 1px solid var(--border);
            padding: 12px;
            border-radius: 6px;
            display: flex;
            flex-direction: column;
            gap: 4px;
        }

        .metric-label {
            font-size: 0.72rem;
            color: var(--text-muted);
            text-transform: uppercase;
            font-weight: 600;
            letter-spacing: 0.04em;
        }

        .metric-value {
            font-size: 1.15rem;
            font-weight: 700;
            color: var(--primary);
            font-family: var(--font-mono);
        }

        .metric-caption {
            font-size: 0.72rem;
            color: var(--text-muted);
        }

        /* Code/JSON Inspector */
        pre {
            background: var(--bg-body);
            border: 1px solid var(--border);
            padding: 12px;
            border-radius: 6px;
            font-family: var(--font-mono);
            font-size: 0.82rem;
            color: #7ee787;
            overflow-x: auto;
            max-height: 380px;
        }

        /* Table styles */
        table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.84rem;
        }

        th, td {
            padding: 10px 12px;
            text-align: left;
            border-bottom: 1px solid var(--border);
        }

        th {
            background: var(--bg-card);
            color: var(--text-muted);
            font-weight: 600;
        }

        tr:hover td {
            background: rgba(255, 255, 255, 0.02);
        }

        .spinner {
            display: inline-block;
            width: 14px;
            height: 14px;
            border: 2px solid rgba(13,17,23,0.3);
            border-radius: 50%;
            border-top-color: #0d1117;
            animation: spin 0.8s linear infinite;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }
    </style>
</head>
<body>

    <!-- Navbar -->
    <header class="navbar">
        <div class="brand">
            <div class="brand-icon">⚡</div>
            <div>
                <span class="brand-title">Financial RAG</span>
                <span class="brand-badge">FastAPI Console</span>
            </div>
        </div>
        <div class="nav-status">
            <div class="status-pill">
                <span class="status-dot"></span>
                <span>FastAPI :8000</span>
            </div>
            <div class="status-pill">
                <span class="status-dot" style="background:#4285f4; box-shadow:0 0 8px #4285f4;"></span>
                <span>Gemini 2.0 Flash</span>
            </div>
            <a href="/docs" target="_blank" class="nav-btn">Swagger API Docs ↗</a>
        </div>
    </header>

    <!-- Main Container -->
    <main class="main-container">
        <!-- Tab Navigation -->
        <nav class="tabs-header">
            <button class="tab-btn active" id="tab-rag-btn" onclick="switchTab('rag')">🚀 RAG Pipeline Tester</button>
            <button class="tab-btn" id="tab-intent-btn" onclick="switchTab('intent')">🎯 Intent Classifier Lab</button>
            <button class="tab-btn" id="tab-audits-btn" onclick="switchTab('audits')">📋 PostgreSQL Audit Logs</button>
        </nav>

        <!-- TAB 1: RAG Pipeline Tester -->
        <section id="tab-rag" class="tab-content active">
            <div class="console-grid">
                <!-- Left Column: Input & Results -->
                <div class="panel">
                    <div class="panel-title">
                        <span>💬 Query Execution Console</span>
                        <span id="timing-badge" style="font-size: 0.8rem; color: var(--text-muted); font-family: var(--font-mono);"></span>
                    </div>

                    <!-- Quick Prompt Presets -->
                    <div>
                        <div style="font-size: 0.78rem; color: var(--text-muted); margin-bottom: 6px; font-weight: 500;">QUICK TEST PRESETS:</div>
                        <div class="quick-prompts">
                            <button class="chip-btn" onclick="setQuery('What was Apple\\'s total iPhone net sales in Q2 2024 compared to Q2 2023?')">📊 Apple Q2 iPhone</button>
                            <button class="chip-btn" onclick="setQuery('What were Amazon AWS net sales and growth rates in 2024?')">☁️ Amazon AWS</button>
                            <button class="chip-btn" onclick="setQuery('Show Google stock performance graph and cumulative returns')">🖼️ Google Stock Chart</button>
                            <button class="chip-btn" onclick="setQuery('Compare Apple and Google revenue 2024')">🔀 Apple vs Google</button>
                            <button class="chip-btn" onclick="setQuery('What was Tesla\\'s net income in Q3 2024?')">🛡️ Tesla (Out of Scope)</button>
                            <button class="chip-btn" onclick="setQuery('Hello, what capabilities do you have?')">👋 Greeting Chitchat</button>
                        </div>
                    </div>

                    <!-- Input Area -->
                    <div class="query-input-box">
                        <textarea id="query-input" placeholder="Ask SEC financial question (e.g., Apple iPhone sales, AWS cloud revenue, charts)..."></textarea>
                        <div class="action-row">
                            <div style="display: flex; align-items: center; gap: 12px; flex-wrap: wrap;">
                                <div style="display: flex; align-items: center; gap: 6px;">
                                    <label style="font-size: 0.8rem; color: var(--text-muted);">LLM Engine:</label>
                                    <select id="provider-select" style="background: var(--bg-card); color: var(--text); border: 1px solid var(--border); padding: 4px 8px; border-radius: 4px; font-family: inherit;">
                                        <option value="gemini" selected>Google Gemini 3.6 Flash</option>
                                        <option value="mistral">Mistral AI (Pixtral 12B Vision)</option>
                                    </select>
                                </div>
                                <div style="display: flex; align-items: center; gap: 6px;">
                                    <label style="font-size: 0.8rem; color: var(--text-muted);">Top-K:</label>
                                    <select id="topk-select" style="background: var(--bg-card); color: var(--text); border: 1px solid var(--border); padding: 4px 8px; border-radius: 4px; font-family: inherit;">
                                        <option value="3" selected>3</option>
                                        <option value="5">5</option>
                                        <option value="8">8</option>
                                    </select>
                                </div>
                            </div>
                            <button class="btn-primary" id="run-query-btn" onclick="runRAGQuery()">
                                <span id="run-btn-text">Execute RAG</span>
                            </button>
                        </div>
                    </div>

                    <!-- Live Intent Badge -->
                    <div class="intent-badge-card" id="intent-card" style="display: none;">
                        <div>
                            <span class="badge-pill" id="intent-badge-pill">📊 Numerical Analysis</span>
                            <span style="font-size: 0.8rem; color: var(--text-muted); margin-left: 8px;" id="intent-conf-text">Confidence: 95%</span>
                        </div>
                        <div style="font-size: 0.78rem; color: var(--text-muted);" id="intent-entity-text">Entity: Apple</div>
                    </div>

                    <!-- Generated Answer Output -->
                    <div>
                        <div style="font-size: 0.82rem; font-weight: 600; margin-bottom: 6px;">Verified Financial Answer:</div>
                        <div class="answer-box" id="answer-output">Run a query to inspect the generated response, retrieved filing citations, and fact-checking metrics...</div>
                    </div>

                    <!-- Visual Chart Images (if any) -->
                    <div id="chart-display-container" style="display: none;">
                        <div style="font-size: 0.82rem; font-weight: 600; margin-bottom: 6px;">📊 Visual SEC Chart Evidence:</div>
                        <div id="chart-img-wrapper" style="border: 1px solid var(--border); border-radius: 6px; padding: 8px; background: var(--bg-card); text-align: center;"></div>
                    </div>
                </div>

                <!-- Right Column: Telemetry Scorecard & Sources -->
                <div class="panel">
                    <div class="panel-title">
                        <span>📊 Real-Time Metrics Scorecard</span>
                        <span id="audit-id-badge" style="font-size: 0.78rem; color: var(--success); font-family: var(--font-mono);"></span>
                    </div>

                    <!-- 6 Production Metrics Grid -->
                    <div class="metrics-grid">
                        <div class="metric-cell">
                            <span class="metric-label">Faithfulness</span>
                            <span class="metric-value" id="m-faith">--%</span>
                            <span class="metric-caption" id="m-faith-sub">Zero Hallucination</span>
                        </div>
                        <div class="metric-cell">
                            <span class="metric-label">Relevance</span>
                            <span class="metric-value" id="m-rel">--%</span>
                            <span class="metric-caption" id="m-rel-sub">Context & Answer</span>
                        </div>
                        <div class="metric-cell">
                            <span class="metric-label">Recall</span>
                            <span class="metric-value" id="m-rec">--%</span>
                            <span class="metric-caption" id="m-rec-sub">Entity Coverage</span>
                        </div>
                        <div class="metric-cell">
                            <span class="metric-label">Coherence</span>
                            <span class="metric-value" id="m-coh">--</span>
                            <span class="metric-caption" id="m-coh-sub">Structure & Flow</span>
                        </div>
                        <div class="metric-cell">
                            <span class="metric-label">Citations</span>
                            <span class="metric-value" id="m-cit">--%</span>
                            <span class="metric-caption" id="m-cit-sub">Provenance</span>
                        </div>
                        <div class="metric-cell">
                            <span class="metric-label">Cost & Latency</span>
                            <span class="metric-value" id="m-cost" style="font-size: 0.95rem;">$0.00</span>
                            <span class="metric-caption" id="m-lat">0.00s</span>
                        </div>
                    </div>

                    <!-- Verified Sources Table -->
                    <div style="margin-top: 8px;">
                        <div style="font-size: 0.85rem; font-weight: 600; margin-bottom: 8px;">📋 Verified SEC Filing Sources:</div>
                        <div style="max-height: 240px; overflow-y: auto; border: 1px solid var(--border); border-radius: 6px;">
                            <table id="sources-table">
                                <thead>
                                    <tr>
                                        <th>Company</th>
                                        <th>Filing / Page</th>
                                        <th>Rerank Score</th>
                                        <th>Preview</th>
                                    </tr>
                                </thead>
                                <tbody id="sources-body">
                                    <tr>
                                        <td colspan="4" style="text-align: center; color: var(--text-muted); padding: 20px;">No active query sources</td>
                                    </tr>
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>
            </div>
        </section>

        <!-- TAB 2: Intent Classifier Lab -->
        <section id="tab-intent" class="tab-content">
            <div class="panel">
                <div class="panel-title">
                    <span>🎯 Intent Classifier Lab (Zero-Vector Instant Test)</span>
                </div>
                <p style="font-size: 0.88rem; color: var(--text-muted);">
                    Test the Financial Intent Classifier independently without invoking Qdrant Cloud or Gemini LLM. Evaluates entity extraction, year/quarter parsing, and routing decisions.
                </p>

                <div class="query-input-box" style="margin-top: 10px;">
                    <textarea id="intent-lab-input" placeholder="Type query to test classification (e.g., 'Compare Meta and Amazon operating margin', 'Hi', 'Netflix stock forecast')..."></textarea>
                    <div class="action-row">
                        <div style="font-size: 0.82rem; color: var(--text-muted);">Latency: &lt; 0.001s</div>
                        <button class="btn-primary" onclick="testIntentOnly()">Classify Intent Only</button>
                    </div>
                </div>

                <div style="margin-top: 14px;">
                    <div style="font-size: 0.85rem; font-weight: 600; margin-bottom: 6px;">Parsed Classification JSON:</div>
                    <pre id="intent-json-output">// Click 'Classify Intent Only' to view full JSON payload...</pre>
                </div>
            </div>
        </section>

        <!-- TAB 3: PostgreSQL Audit Logs -->
        <section id="tab-audits" class="tab-content">
            <div class="panel">
                <div class="panel-title">
                    <span>📋 Live PostgreSQL RAG Audit Logs</span>
                    <button class="btn-secondary" onclick="fetchRecentAudits()">🔄 Refresh Logs</button>
                </div>
                <p style="font-size: 0.88rem; color: var(--text-muted);">
                    Real-time query telemetry records queried from <code>rag_audit_logs</code> table in PostgreSQL.
                </p>

                <div style="overflow-x: auto; margin-top: 12px; border: 1px solid var(--border); border-radius: 6px;">
                    <table>
                        <thead>
                            <tr>
                                <th>Timestamp</th>
                                <th>Intent</th>
                                <th>Company</th>
                                <th>Faithfulness</th>
                                <th>Recall</th>
                                <th>Latency</th>
                                <th>Cost</th>
                                <th>Fallback</th>
                            </tr>
                        </thead>
                        <tbody id="audits-body">
                            <tr>
                                <td colspan="8" style="text-align: center; color: var(--text-muted); padding: 24px;">Click 'Refresh Logs' to load PostgreSQL audit logs</td>
                            </tr>
                        </tbody>
                    </table>
                </div>
            </div>
        </section>
    </main>

    <script>
        function switchTab(tabId) {
            document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(sec => sec.classList.remove('active'));

            document.getElementById('tab-' + tabId + '-btn').classList.add('active');
            document.getElementById('tab-' + tabId).classList.add('active');

            if (tabId === 'audits') {
                fetchRecentAudits();
            }
        }

        function setQuery(text) {
            document.getElementById('query-input').value = text;
        }

        async function runRAGQuery() {
            const query = document.getElementById('query-input').value.trim();
            if (!query) {
                alert('Please enter a query to test.');
                return;
            }

            const topK = parseInt(document.getElementById('topk-select').value) || 3;
            const provider = document.getElementById('provider-select').value || 'gemini';
            const runBtn = document.getElementById('run-query-btn');
            const runText = document.getElementById('run-btn-text');
            const timingBadge = document.getElementById('timing-badge');
            const answerBox = document.getElementById('answer-output');
            const intentCard = document.getElementById('intent-card');
            const chartContainer = document.getElementById('chart-display-container');

            runBtn.disabled = true;
            runText.innerHTML = '<span class="spinner"></span> Running...';
            timingBadge.innerText = 'Analyzing...';
            answerBox.innerText = `⚡ Retrieving SEC chunks from Qdrant Cloud and generating with ${provider.toUpperCase()}...`;

            const startTime = performance.now();

            try {
                const resp = await fetch('/api/query', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ query: query, top_k: topK, provider: provider })
                });

                if (!resp.ok) {
                    const err = await resp.json();
                    throw new Error(err.detail || 'Query failed');
                }

                const data = await resp.json();
                const clientLatency = ((performance.now() - startTime) / 1000).toFixed(2);
                timingBadge.innerText = `⏱️ Server: ${data.latency_seconds}s | Total: ${clientLatency}s`;

                // Render Intent Badge
                if (data.intent) {
                    intentCard.style.display = 'flex';
                    const pill = document.getElementById('intent-badge-pill');
                    pill.innerText = data.intent.badge_label;
                    pill.style.color = data.intent.badge_color;
                    pill.style.background = data.intent.badge_color + '18';
                    pill.style.borderColor = data.intent.badge_color + '40';

                    document.getElementById('intent-conf-text').innerText = `Confidence: ${Math.round(data.intent.confidence * 100)}%`;
                    const entities = data.intent.companies && data.intent.companies.length > 0 
                        ? data.intent.companies.join(', ') 
                        : (data.intent.company || 'N/A');
                    document.getElementById('intent-entity-text').innerText = `Entities: ${entities}`;
                }

                // Render Answer
                answerBox.innerText = data.answer;

                // Render Charts if attached
                if (data.charts && data.charts.length > 0) {
                    chartContainer.style.display = 'block';
                    const wrapper = document.getElementById('chart-img-wrapper');
                    wrapper.innerHTML = '';
                    data.charts.forEach(c => {
                        const img = document.createElement('img');
                        img.src = c.url;
                        img.alt = c.title;
                        img.style.maxWidth = '100%';
                        img.style.maxHeight = '320px';
                        img.style.borderRadius = '6px';
                        wrapper.appendChild(img);
                        const caption = document.createElement('div');
                        caption.style.fontSize = '0.78rem';
                        caption.style.color = '#8b949e';
                        caption.style.marginTop = '6px';
                        caption.innerText = `📊 ${c.title} (${c.filename}, Page ${c.page})`;
                        wrapper.appendChild(caption);
                    });
                } else {
                    chartContainer.style.display = 'none';
                }

                // Render Metrics
                if (data.metrics) {
                    const m = data.metrics;
                    document.getElementById('m-faith').innerText = m.faithfulness ? m.faithfulness.percentage : '--%';
                    document.getElementById('m-rel').innerText = m.relevance ? m.relevance.percentage : '--%';
                    document.getElementById('m-rec').innerText = m.recall ? m.recall.percentage : '--%';
                    document.getElementById('m-coh').innerText = m.coherence ? (m.coherence.score ? m.coherence.score.toFixed(2) : '0.90') : '--';
                    document.getElementById('m-cit').innerText = m.citation_accuracy ? m.citation_accuracy.percentage : '--%';
                    document.getElementById('m-cost').innerText = m.cost ? m.cost.formatted_cost : '$0.00';
                    document.getElementById('m-lat').innerText = `${m.latency_seconds ? m.latency_seconds.toFixed(2) : data.latency_seconds}s`;
                }

                if (data.audit_id) {
                    document.getElementById('audit-id-badge').innerText = `PostgreSQL Audit ID: #${data.audit_id}`;
                }

                // Render Sources
                const tbody = document.getElementById('sources-body');
                tbody.innerHTML = '';
                if (data.sources && data.sources.length > 0) {
                    data.sources.forEach(s => {
                        const tr = document.createElement('tr');
                        tr.innerHTML = `
                            <td><strong style="color:var(--primary);">${s.company || 'N/A'}</strong></td>
                            <td><code>${s.filename || ''}</code> (p.${s.page || '?'})</td>
                            <td><span style="font-family:var(--font-mono); color:var(--success);">${s.rerank_score}</span></td>
                            <td style="font-size:0.78rem; color:var(--text-muted);">${s.content_preview}</td>
                        `;
                        tbody.appendChild(tr);
                    });
                } else {
                    tbody.innerHTML = '<tr><td colspan="4" style="text-align:center; color:var(--text-muted); padding:16px;">Direct response (zero vector chunks needed)</td></tr>';
                }

            } catch (err) {
                answerBox.innerText = '❌ Error executing query: ' + err.message;
            } finally {
                runBtn.disabled = false;
                runText.innerText = 'Execute RAG';
            }
        }

        async function testIntentOnly() {
            const query = document.getElementById('intent-lab-input').value.trim();
            if (!query) {
                alert('Please enter a query in the Intent Lab box.');
                return;
            }
            const out = document.getElementById('intent-json-output');
            out.innerText = 'Classifying...';
            try {
                const resp = await fetch('/api/classify', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ query: query })
                });
                const data = await resp.json();
                out.innerText = JSON.stringify(data, null, 2);
            } catch (err) {
                out.innerText = 'Error: ' + err.message;
            }
        }

        async function fetchRecentAudits() {
            const tbody = document.getElementById('audits-body');
            tbody.innerHTML = '<tr><td colspan="8" style="text-align:center; color:var(--text-muted); padding:16px;">Loading latest PostgreSQL audits...</td></tr>';
            try {
                const resp = await fetch('/api/audits?limit=10');
                const logs = await resp.json();
                tbody.innerHTML = '';
                if (!logs || logs.length === 0) {
                    tbody.innerHTML = '<tr><td colspan="8" style="text-align:center; color:var(--text-muted); padding:16px;">No audit logs found in PostgreSQL.</td></tr>';
                    return;
                }
                logs.forEach(l => {
                    const tr = document.createElement('tr');
                    tr.innerHTML = `
                        <td style="font-family:var(--font-mono); font-size:0.78rem;">${l.timestamp}</td>
                        <td><span style="background:rgba(88,166,255,0.12); color:#58a6ff; padding:2px 8px; border-radius:10px; font-size:0.75rem; font-weight:600;">${l.intent || 'FINANCIAL'}</span></td>
                        <td><strong>${l.company}</strong></td>
                        <td style="color:var(--success); font-family:var(--font-mono);">${l.faithfulness}</td>
                        <td style="font-family:var(--font-mono);">${l.recall}</td>
                        <td style="font-family:var(--font-mono);">${l.latency}</td>
                        <td style="font-family:var(--font-mono);">${l.cost}</td>
                        <td>${l.fallback === 'Yes' ? '⚠️ Yes' : '✅ No'}</td>
                    `;
                    tbody.appendChild(tr);
                });
            } catch (err) {
                tbody.innerHTML = `<tr><td colspan="8" style="color:var(--danger); text-align:center; padding:16px;">Failed to fetch audits: ${err.message}</td></tr>`;
            }
        }
    </script>
</body>
</html>"""
    return HTMLResponse(content=html_content)


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    print(f"🚀 Starting FastAPI Testing Server on http://localhost:{port}")
    uvicorn.run("api:app", host="0.0.0.0", port=port, reload=True)
