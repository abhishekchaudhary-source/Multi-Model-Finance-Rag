"""
Enterprise PostgreSQL Persistence & Analytics Layer for Financial RAG.
Provides connection pooling, chat history persistence, and audit logging.
"""

import os
import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv
from sqlalchemy import (
    create_engine, Column, Integer, Float, String, Text, Boolean, DateTime
)
from sqlalchemy.orm import declarative_base, sessionmaker, scoped_session

load_dotenv()
logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:Mob%401234@localhost:5432/finance_rag"
)

Base = declarative_base()


class RAGAuditLog(Base):
    """Stores query execution telemetry, latency, cost, and 6 core evaluation metrics."""
    __tablename__ = "rag_audit_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    session_id = Column(String(64), index=True, default="default-session")
    query = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    company = Column(String(64), nullable=True)
    faithfulness_score = Column(Float, default=0.0)
    relevance_score = Column(Float, default=0.0)
    recall_score = Column(Float, default=0.0)
    coherence_score = Column(Float, default=0.0)
    citation_accuracy = Column(Float, default=0.0)
    cost_usd = Column(Float, default=0.0)
    latency_seconds = Column(Float, default=0.0)
    is_fallback = Column(Boolean, default=False)
    top_source = Column(String(255), nullable=True)


class ChatMessageRecord(Base):
    """Stores persistent multi-turn chat messages across user sessions."""
    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(64), index=True, default="default-session")
    role = Column(String(32), nullable=False)
    content = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    metadata_json = Column(Text, nullable=True)


# Connection Engine & Session Factory with graceful fallback
_engine = None
_SessionFactory = None
_is_connected = False

try:
    _engine = create_engine(
        DATABASE_URL,
        pool_pre_ping=True,
        pool_size=5,
        max_overflow=10,
        connect_args={"connect_timeout": 5}
    )
    _SessionFactory = scoped_session(sessionmaker(bind=_engine))
    Base.metadata.create_all(_engine)
    _is_connected = True
    print("✅ PostgreSQL: Connected to 'finance_rag' database & tables initialized.")
except Exception as e:
    _is_connected = False
    logger.warning(f"⚠️ PostgreSQL connection deferred: {e}")


def is_db_connected() -> bool:
    """Returns True if PostgreSQL is connected and operational."""
    global _is_connected
    if not _engine:
        return False
    try:
        with _engine.connect() as conn:
            return True
    except Exception:
        return False


def init_db():
    """Initializes tables in PostgreSQL database."""
    global _is_connected, _engine, _SessionFactory
    if not _engine:
        try:
            _engine = create_engine(DATABASE_URL, pool_pre_ping=True)
            _SessionFactory = scoped_session(sessionmaker(bind=_engine))
        except Exception as e:
            logger.error(f"Failed to create engine: {e}")
            return False
    try:
        Base.metadata.create_all(_engine)
        _is_connected = True
        return True
    except Exception as e:
        logger.error(f"init_db error: {e}")
        return False


def log_query_audit(
    query: str,
    answer: str,
    metrics: Optional[Dict[str, Any]] = None,
    sources: Optional[List[Dict[str, Any]]] = None,
    session_id: str = "default-session",
    is_fallback: bool = False
) -> Optional[int]:
    """
    Inserts a comprehensive telemetry and evaluation record into rag_audit_logs.
    """
    if not _SessionFactory:
        return None

    session = _SessionFactory()
    try:
        m = metrics or {}
        f_score = m.get("faithfulness", {}).get("score", 0.0)
        rel_score = m.get("relevance", {}).get("score", 0.0)
        rec_score = m.get("recall", {}).get("score", 0.0)
        coh_score = m.get("coherence", {}).get("score", 0.0)
        cit_score = m.get("citation_accuracy", {}).get("score", 0.0)
        cost_val = m.get("cost", {}).get("estimated_cost_usd", 0.0)
        latency_val = m.get("latency_seconds", 0.0)

        top_src = ""
        company_name = ""
        if sources and len(sources) > 0:
            top_src = f"{sources[0].get('filename', '')} (p.{sources[0].get('page', '')})"
            company_name = sources[0].get("company", "")

        record = RAGAuditLog(
            session_id=session_id,
            query=query,
            answer=answer,
            company=company_name,
            faithfulness_score=f_score,
            relevance_score=rel_score,
            recall_score=rec_score,
            coherence_score=coh_score,
            citation_accuracy=cit_score,
            cost_usd=cost_val,
            latency_seconds=latency_val,
            is_fallback=is_fallback,
            top_source=top_src
        )
        session.add(record)
        session.commit()
        return record.id
    except Exception as e:
        session.rollback()
        logger.warning(f"Failed to log query audit to PostgreSQL: {e}")
        return None
    finally:
        session.close()


def save_chat_message(
    role: str,
    content: str,
    session_id: str = "default-session",
    metadata: Optional[Dict[str, Any]] = None
) -> Optional[int]:
    """
    Persists a chat message in PostgreSQL chat_messages table.
    """
    if not _SessionFactory:
        return None

    session = _SessionFactory()
    try:
        meta_str = json.dumps(metadata, default=str) if metadata else None
        record = ChatMessageRecord(
            session_id=session_id,
            role=role,
            content=content,
            metadata_json=meta_str
        )
        session.add(record)
        session.commit()
        return record.id
    except Exception as e:
        session.rollback()
        logger.warning(f"Failed to save chat message: {e}")
        return None
    finally:
        session.close()


def get_chat_history(session_id: str = "default-session", limit: int = 50) -> List[Dict[str, Any]]:
    """
    Retrieves persisted chat message history from PostgreSQL.
    """
    if not _SessionFactory:
        return []

    session = _SessionFactory()
    try:
        records = (
            session.query(ChatMessageRecord)
            .filter(ChatMessageRecord.session_id == session_id)
            .order_by(ChatMessageRecord.timestamp.asc())
            .limit(limit)
            .all()
        )
        return [
            {
                "id": r.id,
                "role": r.role,
                "content": r.content,
                "timestamp": r.timestamp.strftime("%Y-%m-%d %H:%M:%S") if r.timestamp else "",
                "metadata": json.loads(r.metadata_json) if r.metadata_json else {}
            }
            for r in records
        ]
    except Exception as e:
        logger.warning(f"Failed to fetch chat history: {e}")
        return []
    finally:
        session.close()


def get_recent_audit_logs(limit: int = 15) -> List[Dict[str, Any]]:
    """
    Retrieves latest query audits for dashboard inspection.
    """
    if not _SessionFactory:
        return []

    session = _SessionFactory()
    try:
        records = (
            session.query(RAGAuditLog)
            .order_by(RAGAuditLog.timestamp.desc())
            .limit(limit)
            .all()
        )
        return [
            {
                "id": r.id,
                "timestamp": r.timestamp.strftime("%Y-%m-%d %H:%M:%S") if r.timestamp else "",
                "query": r.query,
                "company": r.company or "N/A",
                "faithfulness": f"{r.faithfulness_score * 100:.1f}%",
                "recall": f"{r.recall_score * 100:.1f}%",
                "relevance": f"{r.relevance_score * 100:.1f}%",
                "latency": f"{r.latency_seconds:.2f}s",
                "cost": f"${r.cost_usd:.5f}",
                "fallback": "Yes" if r.is_fallback else "No"
            }
            for r in records
        ]
    except Exception as e:
        logger.warning(f"Failed to fetch audit logs: {e}")
        return []
    finally:
        session.close()


def get_analytics_summary() -> Dict[str, Any]:
    """
    Calculates aggregated analytics across all past queries in PostgreSQL.
    """
    if not _SessionFactory:
        return {
            "total_queries": 0,
            "avg_latency": 0.0,
            "avg_faithfulness": "0.0%",
            "avg_recall": "0.0%",
            "total_cost": "$0.0000"
        }

    session = _SessionFactory()
    try:
        from sqlalchemy import func
        stats = session.query(
            func.count(RAGAuditLog.id),
            func.avg(RAGAuditLog.latency_seconds),
            func.avg(RAGAuditLog.faithfulness_score),
            func.avg(RAGAuditLog.recall_score),
            func.sum(RAGAuditLog.cost_usd)
        ).first()

        count, avg_lat, avg_faith, avg_rec, total_cost = stats
        return {
            "total_queries": count or 0,
            "avg_latency": round(avg_lat or 0.0, 2),
            "avg_faithfulness": f"{(avg_faith or 0.0) * 100:.1f}%",
            "avg_recall": f"{(avg_rec or 0.0) * 100:.1f}%",
            "total_cost": f"${(total_cost or 0.0):.5f}"
        }
    except Exception as e:
        logger.warning(f"Failed to get analytics summary: {e}")
        return {
            "total_queries": 0,
            "avg_latency": 0.0,
            "avg_faithfulness": "0.0%",
            "avg_recall": "0.0%",
            "total_cost": "$0.0000"
        }
    finally:
        session.close()
