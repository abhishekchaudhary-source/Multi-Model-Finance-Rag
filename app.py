import os
import sys
import re
import html
import time
import uuid
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = str(Path(__file__).resolve().parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import streamlit as st
from PIL import Image
from dotenv import load_dotenv
from metrics import FinancialRAGMetrics
from database import (
    is_db_connected,
    log_query_audit,
    save_chat_message,
    get_recent_audit_logs,
    get_analytics_summary
)

load_dotenv()

# Streamlit Page Config - Gemini-style Chatbot UI
st.set_page_config(
    page_title="Gemini | Financial Intelligence",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Session State
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]

if "messages" not in st.session_state:
    st.session_state.messages = []

if "recents" not in st.session_state:
    st.session_state.recents = [
        "What was Apple's total iPhone net sales in Q2 2024 compared to Q2 2023?",
        "What were Google's total research and development (R&D) expenses in 2023?",
        "Show Google stock performance graph and cumulative returns",
        "What were Amazon AWS net sales and growth rates in 2024?"
    ]

if "theme" not in st.session_state:
    st.session_state.theme = "dark"

# --- DYNAMIC THEME SYSTEM (DARK & LIGHT MODES) ---
is_light = (st.session_state.theme == "light")

if is_light:
    css_vars = """
    :root {
        --app-bg: #ffffff;
        --app-text: #1f1f1f;
        --heading-color: #000000;
        --sidebar-bg: #f8f9fa;
        --sidebar-border: rgba(0, 0, 0, 0.08);
        --sidebar-title: #444746;
        --sidebar-text: #1f1f1f;
        --sidebar-hover: #e9eef6;
        --logo-color: #1f1f1f;
        --logo-tag-bg: #e9eef6;
        --logo-tag-color: #0b57d0;
        --user-bubble-bg: #e9eef6;
        --user-bubble-text: #1f1f1f;
        --user-bubble-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
        --ai-avatar-bg: #f0f4f9;
        --ai-avatar-border: rgba(0, 0, 0, 0.12);
        --ai-text: #1f1f1f;
        --pill-bg: #f0f4f9;
        --pill-border: rgba(0, 0, 0, 0.12);
        --pill-text: #1f1f1f;
        --input-bg: #f0f4f9;
        --input-border: rgba(0, 0, 0, 0.15);
        --input-shadow: 0 4px 16px rgba(0, 0, 0, 0.08);
        --input-text: #1f1f1f;
        --btn-bg: #ffffff;
        --btn-border: rgba(0, 0, 0, 0.14);
        --btn-text: #1f1f1f;
        --btn-hover-bg: #e9eef6;
        --btn-hover-border: #a8c7fa;
        --btn-hover-text: #0b57d0;
        --expander-header: #f0f4f9;
        --expander-body: #ffffff;
        --card-bg: #f8fafd;
        --card-border: rgba(0, 0, 0, 0.08);
        --metric-label: #444746;
        --metric-val: #000000;
        --metric-sub: #5e5e5e;
        --code-bg: #e9eef6;
        --code-text: #0b57d0;
        --code-border: rgba(11, 87, 208, 0.2);
        --table-th-bg: #f0f4f9;
        --table-th-color: #0b57d0;
        --table-td-bg: #ffffff;
        --table-td-color: #1f1f1f;
        --table-border: rgba(0, 0, 0, 0.08);
        --disclaimer-color: #5e5e5e;
        --footer-bg: #ffffff;
        --footer-border: rgba(0, 0, 0, 0.08);
        --footer-text: #1f1f1f;
        --footer-subtext: #5e5e5e;
    }
    """
else:
    css_vars = """
    :root {
        --app-bg: #131314;
        --app-text: #e3e3e3;
        --heading-color: #ffffff;
        --sidebar-bg: #1e1f20;
        --sidebar-border: rgba(255, 255, 255, 0.05);
        --sidebar-title: #8e918f;
        --sidebar-text: #c4c7c5;
        --sidebar-hover: #282a2c;
        --logo-color: #ffffff;
        --logo-tag-bg: #282a2c;
        --logo-tag-color: #8ab4f8;
        --user-bubble-bg: #282a2c;
        --user-bubble-text: #ffffff;
        --user-bubble-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
        --ai-avatar-bg: #1e1f20;
        --ai-avatar-border: rgba(255, 255, 255, 0.1);
        --ai-text: #e3e3e3;
        --pill-bg: #1e1f20;
        --pill-border: rgba(255, 255, 255, 0.1);
        --pill-text: #c4c7c5;
        --input-bg: #1e1f20;
        --input-border: rgba(255, 255, 255, 0.15);
        --input-shadow: 0 8px 24px rgba(0, 0, 0, 0.4);
        --input-text: #e3e3e3;
        --btn-bg: #1e1f20;
        --btn-border: rgba(255, 255, 255, 0.1);
        --btn-text: #e3e3e3;
        --btn-hover-bg: #282a2c;
        --btn-hover-border: rgba(255, 255, 255, 0.25);
        --btn-hover-text: #ffffff;
        --expander-header: #1e1f20;
        --expander-body: #18191a;
        --card-bg: #1e1f20;
        --card-border: rgba(255, 255, 255, 0.08);
        --metric-label: #8e918f;
        --metric-val: #ffffff;
        --metric-sub: #8e918f;
        --code-bg: #282a2c;
        --code-text: #8ab4f8;
        --code-border: rgba(138, 180, 248, 0.2);
        --table-th-bg: #1e1f20;
        --table-th-color: #8ab4f8;
        --table-td-bg: #18191a;
        --table-td-color: #e3e3e3;
        --table-border: rgba(255, 255, 255, 0.08);
        --disclaimer-color: #8e918f;
        --footer-bg: #1e1f20;
        --footer-border: rgba(255, 255, 255, 0.05);
        --footer-text: #ffffff;
        --footer-subtext: #8e918f;
    }
    """

# Custom Theme CSS injection
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Google+Sans:wght@400;500;700&family=Inter:wght@300;400;500;600;700&display=swap');
    
    {css_vars}

    html, body, [class*="css"] {{
        font-family: 'Google Sans', 'Inter', sans-serif;
    }}
    
    /* Dynamic Main Application Canvas */
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {{
        background-color: var(--app-bg) !important;
        color: var(--app-text) !important;
    }}

    /* Top Streamlit Header Bar */
    header[data-testid="stHeader"] {{
        background-color: var(--app-bg) !important;
        border-bottom: 1px solid var(--sidebar-border) !important;
    }}
    header[data-testid="stHeader"] * {{
        color: var(--app-text) !important;
    }}

    /* Bottom Chat Input Bar Wrapper */
    div[data-testid="stBottom"], div[data-testid="stBottom"] > div {{
        background-color: var(--app-bg) !important;
    }}
    
    /* Left Sidebar Styling */
    section[data-testid="stSidebar"], section[data-testid="stSidebar"] > div {{
        background-color: var(--sidebar-bg) !important;
        border-right: 1px solid var(--sidebar-border) !important;
        padding-top: 1rem;
    }}
    section[data-testid="stSidebar"] * {{
        color: var(--sidebar-text) !important;
    }}
    
    /* Gemini Logo & Header */
    .gemini-logo {{
        display: flex;
        align-items: center;
        gap: 10px;
        font-size: 20px;
        font-weight: 600;
        color: var(--logo-color) !important;
        margin-bottom: 20px;
        padding: 0 4px;
    }}
    
    .sparkle-icon {{
        font-size: 22px;
        background: linear-gradient(135deg, #4285F4 0%, #9B72CF 50%, #D96570 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }}

    .gemini-tag {{
        font-size: 11px;
        background: var(--logo-tag-bg) !important;
        padding: 2px 8px;
        border-radius: 6px;
        margin-left: 8px;
        color: var(--logo-tag-color) !important;
        font-weight: 500;
    }}

    /* Sidebar Navigation Items */
    .sidebar-section-title {{
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: var(--sidebar-title) !important;
        margin: 16px 0 8px 6px;
        font-weight: 600;
    }}

    /* Streamlit Button Overrides (Sidebar & Main Canvas) */
    div[data-testid="stButton"] button, 
    button[data-testid*="stBaseButton"] {{
        background-color: var(--btn-bg) !important;
        border: 1px solid var(--btn-border) !important;
        color: var(--btn-text) !important;
        border-radius: 10px !important;
        font-size: 13px !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04) !important;
        transition: all 0.15s ease !important;
    }}

    div[data-testid="stButton"] button:hover, 
    button[data-testid*="stBaseButton"]:hover {{
        background-color: var(--btn-hover-bg) !important;
        border-color: var(--btn-hover-border) !important;
        color: var(--btn-hover-text) !important;
    }}

    /* Force all inner elements inside buttons to obey button text color */
    div[data-testid="stButton"] button *, 
    button[data-testid*="stBaseButton"] * {{
        color: var(--btn-text) !important;
    }}

    div[data-testid="stButton"] button:hover *, 
    button[data-testid*="stBaseButton"]:hover * {{
        color: var(--btn-hover-text) !important;
    }}

    /* Bottom User Profile in Sidebar */
    .sidebar-user-footer {{
        position: fixed;
        bottom: 12px;
        left: 12px;
        width: 280px;
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 8px 12px;
        border-radius: 12px;
        background: var(--footer-bg) !important;
        border: 1px solid var(--footer-border) !important;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.04);
    }}
    
    .user-avatar {{
        width: 34px;
        height: 34px;
        border-radius: 50%;
        background: linear-gradient(135deg, #10b981 0%, #3b82f6 100%);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 14px;
        font-weight: 600;
        color: #ffffff !important;
    }}

    .footer-user-name {{
        font-size: 13px;
        font-weight: 600;
        color: var(--footer-text) !important;
    }}

    .footer-user-sub {{
        font-size: 11px;
        color: var(--footer-subtext) !important;
    }}

    /* User Message Bubble */
    .user-msg-row {{
        display: flex;
        justify-content: flex-end;
        margin: 20px 0 14px 0;
    }}
    
    .user-msg-bubble {{
        background-color: var(--user-bubble-bg) !important;
        color: var(--user-bubble-text) !important;
        border: 1px solid var(--card-border) !important;
        border-radius: 18px !important;
        padding: 12px 20px !important;
        max-width: 80% !important;
        font-size: 15px !important;
        line-height: 1.5 !important;
        box-shadow: var(--user-bubble-shadow) !important;
    }}

    .user-msg-bubble * {{
        color: var(--user-bubble-text) !important;
    }}

    /* AI Response Sparkle Avatar */
    .ai-sparkle-avatar {{
        width: 28px;
        height: 28px;
        min-width: 28px;
        border-radius: 50%;
        background: var(--ai-avatar-bg) !important;
        border: 1px solid var(--ai-avatar-border) !important;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 14px;
        color: #4285F4 !important;
    }}

    /* Model Pill Header */
    .model-header-pill {{
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: var(--pill-bg) !important;
        border: 1px solid var(--pill-border) !important;
        padding: 5px 14px;
        border-radius: 9999px;
        font-size: 13px;
        color: var(--pill-text) !important;
        font-weight: 500;
    }}

    /* Floating Chat Input Area */
    .stChatInput {{
        position: fixed;
        bottom: 24px;
        left: 50%;
        transform: translateX(-50%);
        width: 860px !important;
        max-width: 90%;
        z-index: 999;
    }}
    
    .stChatInput > div {{
        background-color: var(--input-bg) !important;
        border: 1px solid var(--input-border) !important;
        border-radius: 28px !important;
        box-shadow: var(--input-shadow) !important;
    }}

    .stChatInput textarea, .stChatInput input {{
        color: var(--input-text) !important;
        background-color: transparent !important;
    }}

    .stChatInput textarea::placeholder {{
        color: var(--disclaimer-color) !important;
    }}

    /* Expanders (Verified Sources & Real-Time RAG Scorecard) */
    div[data-testid="stExpander"] {{
        background-color: var(--expander-body) !important;
        border: 1px solid var(--card-border) !important;
        border-radius: 12px !important;
        margin-top: 10px !important;
        overflow: hidden !important;
    }}

    div[data-testid="stExpander"] details {{
        background-color: var(--expander-body) !important;
    }}

    div[data-testid="stExpander"] summary {{
        background-color: var(--expander-header) !important;
        color: var(--app-text) !important;
        border-bottom: 1px solid var(--card-border) !important;
        font-weight: 500 !important;
        padding: 10px 14px !important;
    }}

    div[data-testid="stExpander"] summary:hover {{
        background-color: var(--sidebar-hover) !important;
    }}

    div[data-testid="stExpander"] summary * {{
        color: var(--app-text) !important;
        fill: var(--app-text) !important;
    }}

    div[data-testid="stExpanderDetails"] {{
        background-color: var(--expander-body) !important;
        color: var(--app-text) !important;
        padding: 14px !important;
    }}

    /* Scorecard Metric Overrides (Faithfulness, Relevance, Recall, etc.) */
    div[data-testid="stMetric"] {{
        background-color: var(--card-bg) !important;
        border: 1px solid var(--card-border) !important;
        border-radius: 10px !important;
        padding: 10px 14px !important;
    }}

    div[data-testid="stMetricLabel"], 
    div[data-testid="stMetricLabel"] * {{
        color: var(--metric-label) !important;
        font-size: 13px !important;
        font-weight: 600 !important;
    }}

    div[data-testid="stMetricValue"], 
    div[data-testid="stMetricValue"] * {{
        color: var(--metric-val) !important;
        font-size: 26px !important;
        font-weight: 700 !important;
    }}

    div[data-testid="stCaptionContainer"], 
    div[data-testid="stCaptionContainer"] * {{
        color: var(--metric-sub) !important;
        font-size: 12px !important;
    }}

    /* Typography & Markdown Content */
    .stMarkdown, .stMarkdown p, .stMarkdown span, .stMarkdown li, .stMarkdown div {{
        color: var(--app-text) !important;
    }}

    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4 {{
        color: var(--heading-color) !important;
        font-weight: 600 !important;
    }}

    /* Code Blocks & Badges */
    code {{
        background-color: var(--code-bg) !important;
        color: var(--code-text) !important;
        border: 1px solid var(--code-border) !important;
        border-radius: 4px !important;
        padding: 2px 6px !important;
        font-size: 13px !important;
    }}

    /* Markdown Tables */
    table {{
        border: 1px solid var(--table-border) !important;
        background-color: var(--table-td-bg) !important;
        border-radius: 8px !important;
        overflow: hidden !important;
        margin: 12px 0 !important;
    }}

    th {{
        background-color: var(--table-th-bg) !important;
        color: var(--table-th-color) !important;
        padding: 8px 12px !important;
        font-weight: 600 !important;
    }}

    td {{
        background-color: var(--table-td-bg) !important;
        color: var(--table-td-color) !important;
        padding: 8px 12px !important;
        border-top: 1px solid var(--table-border) !important;
    }}

    /* Gemini Disclaimer Footer */
    .gemini-disclaimer {{
        text-align: center;
        font-size: 11px;
        color: var(--disclaimer-color) !important;
        margin-top: 8px;
    }}
</style>
""", unsafe_allow_html=True)


# Lazy-load backend components with Streamlit caching
@st.cache_resource(show_spinner="Connecting to FinMultiModal Engine...")
def load_rag_pipeline():
    from retriever import FinancialMultiModalRerankRetriever
    from generator import GeminiFinancialGenerator
    from rag_engine import FinancialGuardrails, FallbackManager

    retriever = FinancialMultiModalRerankRetriever()
    generator = GeminiFinancialGenerator()
    guardrails = FinancialGuardrails()
    fallback = FallbackManager()
    return retriever, generator, guardrails, fallback


retriever, generator, guardrails, fallback = load_rag_pipeline()


def sanitize_query(raw_query: str) -> str:
    """Cleans up query string by stripping leading numbers and test markers."""
    cleaned = re.sub(r'^\s*\d+[\.\)]\s*', '', raw_query)
    cleaned = cleaned.strip('"\' ')
    cleaned = re.sub(r'\s*\([Tt]ests?[^)]*\)\s*$', '', cleaned)
    return cleaned.strip()


def clean_answer_text(text: str) -> str:
    """
    Strips obsolete tip lines, fallback banners, and redundant query/metadata headers
    so only the pure verified financial answer is displayed.
    """
    # 1. Remove Tip: Add GEMINI_API_KEY line
    text = re.sub(r'>?\s*💡\s*\*?Tip:.*?(?:\n+|$)', '', text, flags=re.IGNORECASE)
    # 2. Remove Direct Verified Filing Evidence banner
    text = re.sub(r'#*\s*📋\s*Direct Verified Filing Evidence.*?(?:\n+|$)', '', text, flags=re.IGNORECASE)
    # 3. Remove Query / Top Verified Source / Reranker Confidence Score block
    text = re.sub(r'\*?\*?Query:\*?\*?.*?(?:\n+|$)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\*?\*?Top Verified Source:\*?\*?.*?(?:\n+|$)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\*?\*?Reranker Confidence Score:\*?\*?.*?(?:\n+|$)', '', text, flags=re.IGNORECASE)
    # 4. Remove Exact Filing Data header
    text = re.sub(r'\*?\*?Exact Filing Data:\*?\*?\s*(?:\n+|$)', '', text, flags=re.IGNORECASE)
    return text.strip()



def start_new_query(user_query: str):
    """
    Appends the new question to conversation history so all questions and answers
    remain visible on screen, while streaming the latest response word-by-word.
    """
    st.session_state.messages.append({"role": "user", "content": user_query})
    if user_query not in st.session_state.recents:
        st.session_state.recents.insert(0, user_query)
    st.session_state.pending_query = user_query
    # Persist question to PostgreSQL
    save_chat_message(role="user", content=user_query, session_id=st.session_state.session_id)




# --- LEFT SIDEBAR (EXACT GEMINI STYLE) ---
with st.sidebar:
    st.markdown("""
    <div class="gemini-logo">
        <span class="sparkle-icon">✦</span>
        <span>Gemini</span>
        <span class="gemini-tag">Finance</span>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar Header: New Chat, Clear, Theme Toggle
    col_new1, col_new2, col_new3 = st.columns([2.6, 1, 1])
    with col_new1:
        if st.button("＋  New chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.pending_query = None
            st.rerun()
    with col_new2:
        if st.button("🗑️", help="Clear current conversation", use_container_width=True):
            st.session_state.messages = []
            st.session_state.pending_query = None
            st.rerun()
    with col_new3:
        theme_icon = "☀️" if st.session_state.theme == "dark" else "🌙"
        theme_tooltip = "Switch to Light Mode" if st.session_state.theme == "dark" else "Switch to Dark Mode"
        if st.button(theme_icon, key="sidebar_theme_toggle", help=theme_tooltip, use_container_width=True):
            st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"
            st.rerun()

    # Sidebar Navigation (Search Chats)
    st.markdown("""
    <div style="margin-top: 10px;">
        <div class="recent-item">🔍 Search chats</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section-title">Recent Inquiries (Click to Ask)</div>', unsafe_allow_html=True)
    
    # Render Recents list
    for idx, recent_title in enumerate(st.session_state.recents[:6]):
        display_label = recent_title[:38] + "..." if len(recent_title) > 38 else recent_title
        if st.button(f"💬 {display_label}", key=f"recent_btn_{idx}", use_container_width=True, help=recent_title):
            start_new_query(recent_title)
            st.rerun()

    # PostgreSQL Persistence & Analytics Expander
    db_active = is_db_connected()
    db_status_badge = "🟢 Connected" if db_active else "🔴 Offline"
    with st.expander(f"🐘 PostgreSQL ({db_status_badge})", expanded=False):
        st.caption(f"**Database:** `finance_rag` | Port `5432`")
        if db_active:
            stats = get_analytics_summary()
            col_db1, col_db2 = st.columns(2)
            with col_db1:
                st.metric("Total Audits", stats["total_queries"])
                st.metric("Avg Recall", stats["avg_recall"])
            with col_db2:
                st.metric("Avg Latency", f"{stats['avg_latency']}s")
                st.metric("Total Cost", stats["total_cost"])

            logs = get_recent_audit_logs(limit=4)
            if logs:
                st.markdown("<div style='font-size:11px; font-weight:600; margin-top:8px;'>Recent Logged Audits:</div>", unsafe_allow_html=True)
                for l in logs:
                    st.caption(f"• `{l['timestamp'][-8:]}` | **{l['company']}** | Faith: `{l['faithfulness']}` | Lat: `{l['latency']}`")
        else:
            st.warning("PostgreSQL connection offline. Connect to localhost:5432 to enable persistence.")

    st.markdown("<br>", unsafe_allow_html=True)

    # User Profile Footer
    st.markdown("""
    <div class="sidebar-user-footer">
        <div class="user-avatar">AC</div>
        <div style="flex: 1; overflow: hidden;">
            <div class="footer-user-name">Abhishek Chaudhary</div>
            <div class="footer-user-sub">Financial Analyst Workspace</div>
        </div>
        <div style="color: var(--sidebar-title); font-size: 14px;">⚙️</div>
    </div>
    """, unsafe_allow_html=True)


# --- MAIN CHAT AREA ---
col_head_left, col_head_right = st.columns([3, 1])
with col_head_left:
    llm_display = os.getenv("LLM_MODEL", "gemini-2.0-flash").replace("-", " ").title()
    st.markdown(f"""
    <div class="model-header-pill">
        <span style="color: #4285F4;">●</span>
        <span>{llm_display}</span>
        <span style="font-size: 10px; opacity: 0.7; margin-left: 4px;">SEC Multi-Modal RAG</span>
    </div>
    """, unsafe_allow_html=True)

with col_head_right:
    db_indicator = '<span style="color: #34A853;">●</span> PG: <b>Active</b>' if is_db_connected() else '<span style="color: #EA4335;">●</span> PG: <b>Off</b>'
    st.markdown(f"""
    <div style="font-size: 12px; opacity: 0.85; text-align: right; padding-top: 6px;">
        Qdrant: <b>8,045</b> | {db_indicator}
    </div>
    """, unsafe_allow_html=True)



# Welcome banner if no messages yet
if not st.session_state.messages:
    welcome_sub_color = "#5e5e5e" if is_light else "#c4c7c5"
    st.markdown(f"""
    <div style="text-align: center; margin: 60px 0 40px 0;">
        <h1 style="font-size: 42px; font-weight: 500; background: linear-gradient(135deg, #4285F4 0%, #9B72CF 50%, #D96570 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
            Hello, Abhishek
        </h1>
        <p style="font-size: 20px; color: {welcome_sub_color}; font-weight: 400; margin-top: 10px;">
            How can I help with Apple, Amazon, Google or Meta financial filings today?
        </p>
    </div>
    """, unsafe_allow_html=True)


# Render Active Questions & Answers (Full Chat History)
for msg in st.session_state.messages:
    if msg["role"] == "user":
        user_escaped = html.escape(msg["content"])
        st.markdown(
            f'<div class="user-msg-row"><div class="user-msg-bubble">💬 {user_escaped}</div></div>',
            unsafe_allow_html=True
        )

    elif msg["role"] == "assistant":
        st.markdown("""
        <div style="display: flex; align-items: center; gap: 8px; margin-top: 14px; margin-bottom: 6px;">
            <div class="ai-sparkle-avatar">✦</div>
            <span style="font-size: 14px; font-weight: 600; color: var(--app-text);">Gemini</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(clean_answer_text(msg["content"]))

        # Display chart image if present in the message state
        if msg.get("chart_images"):
            for img_info in msg["chart_images"]:
                if os.path.exists(img_info["path"]):
                    c1, c2 = st.columns([1.2, 1])
                    with c1:
                        st.image(img_info["path"], caption=f"📊 {img_info['title']}", use_container_width=True)
                    with c2:
                        st.caption(f"📁 **Filing:** `{img_info['filename']}` (Page {img_info['page']})")
                        st.caption(f"🎯 **BGE-Reranker Score:** `{img_info['score']}`")

        # Source Evidence Expander
        if msg.get("sources"):
            with st.expander(f"📋 Verified SEC Sources ({len(msg['sources'])} Chunks)"):
                for idx, src in enumerate(msg["sources"], 1):
                    st.caption(f"**Source {idx}:** `{src['filename']}` | Page {src['page']} | Modality: `{src['modality']}` | Score: `{src['rerank_score']}`")
                    st.markdown(src["content"][:300] + "...")
                    st.markdown("---")

        # Real-Time Metrics & Scorecard Expander (6 Core Production Metrics)
        if msg.get("metrics"):
            m = msg["metrics"]
            is_fb = m.get("is_fallback") or m.get("faithfulness", {}).get("status") == "FALLBACK SAFE"
            scorecard_title = "🛡️ Out-of-Scope Fallback Scorecard (Safe Deflection - Zero Hallucination)" if is_fb else f"📊 Real-Time RAG Metrics & Audit Scorecard (Quality: {m['quality_percentage']})"
            with st.expander(scorecard_title):
                mc1, mc2, mc3 = st.columns(3)
                with mc1:
                    st.metric(label="🛡️ Faithfulness (Groundedness)", value=m["faithfulness"]["percentage"])
                    st.caption(f"Verified entities: {m['faithfulness']['verified_count']} | Status: `{m['faithfulness']['status']}`")
                    st.metric(label="🎯 Relevance (Context & Answer)", value=m["relevance"]["percentage"])
                    st.caption(f"Context: {m['relevance']['context_relevance']*100:.0f}% | Answer: {m['relevance']['answer_relevance']*100:.0f}%")
                with mc2:
                    st.metric(label="🔍 Retrieval Recall", value=m["recall"]["percentage"])
                    st.caption(f"Entities: {', '.join(m['recall']['matched_entities'][:3]) or 'General query'}")
                    st.metric(label="📝 Coherence Score", value=m["coherence"].get("normalized_score", f"{m['coherence'].get('score', 0.9):.2f} / 1.0"))
                    st.caption(f"Scale: 0.0 – 1.0 ({m['coherence'].get('percentage', '90.0%')}) | `{m['coherence'].get('status', 'EXCELLENT')}`")
                with mc3:
                    st.metric(label="📑 Citation Accuracy", value=m["citation_accuracy"]["percentage"])
                    st.caption(f"Valid: {m['citation_accuracy']['valid_citations']}/{m['citation_accuracy']['total_citations']} citations")
                    st.metric(label="💰 Cost Per Query", value=m["cost"]["formatted_cost"])
                    st.caption(f"Tokens: {m['cost']['total_tokens']} | Latency: {m['latency_seconds']:.2f}s")


# Process pending query with real-time word-by-word streaming
if st.session_state.get("pending_query"):
    query_to_run = st.session_state.pop("pending_query")
    cleaned_query = sanitize_query(query_to_run)
    start_time = time.time()

    # Gemini Assistant Header
    st.markdown("""
    <div style="display: flex; align-items: center; gap: 8px; margin-top: 14px; margin-bottom: 6px;">
        <div class="ai-sparkle-avatar">✦</div>
        <span style="font-size: 14px; font-weight: 600; color: var(--app-text);">Gemini</span>
    </div>
    """, unsafe_allow_html=True)

    # 1. Guardrail Check
    input_check = guardrails.validate_input(cleaned_query)
    if not input_check["passed"]:
        sec_msg = f"🛡️ **Security Guardrail:** {input_check['reason']}"
        def stream_sec():
            parts = re.split(r'( +|\n+)', sec_msg)
            for part in parts:
                if part:
                    yield part
                    if part.strip():
                        time.sleep(0.025)
        st.write_stream(stream_sec)
        st.session_state.messages.append({
            "role": "assistant",
            "content": sec_msg,
            "sources": [],
            "chart_images": [],
            "metrics": None
        })
        # Persist guardrail block to PostgreSQL
        save_chat_message(role="assistant", content=sec_msg, session_id=st.session_state.session_id)
        log_query_audit(
            query=cleaned_query,
            answer=sec_msg,
            metrics=None,
            sources=[],
            session_id=st.session_state.session_id,
            is_fallback=True
        )
    else:
        # 2. Retrieval & Generation with status spinner (Question is already visible at the top!)
        with st.spinner("Analyzing SEC filings with Gemini 2.0 Flash..."):
            chunks = retriever.retrieve_and_rerank(
                query=cleaned_query,
                dense_top_k=int(os.getenv("DENSE_TOP_K", 12)),
                sparse_top_k=int(os.getenv("SPARSE_TOP_K", 12)),
                hybrid_top_k=int(os.getenv("HYBRID_TOP_K", 12)),
                top_k=int(os.getenv("RERANK_TOP_K", 3)),
                min_rerank_score=0.20
            )

            # 3. Answer Generation
            if not chunks:
                answer_text = fallback.out_of_scope_fallback(cleaned_query)
                chart_imgs = []
            else:
                gen_result = generator.generate(cleaned_query, chunks)
                answer_text = clean_answer_text(gen_result["answer"])
                chart_imgs = []
                for c in chunks:
                    if c.get("image_path") and os.path.exists(c["image_path"]):
                        chart_imgs.append({
                            "path": c["image_path"],
                            "title": f"{c['company']} {c.get('year', '')} Visual Evidence",
                            "filename": c.get("filename", ""),
                            "page": c.get("page", ""),
                            "score": c.get("rerank_score", "")
                        })

        # 4. Stream Answer Word-by-Word (Typewriter effect: word by word visible typing)
        def stream_word_by_word(text: str):
            parts = re.split(r'( +|\n+)', text)
            for part in parts:
                if part:
                    yield part
                    if part.strip():
                        time.sleep(0.025)

        st.write_stream(stream_word_by_word(answer_text))

        # 5. Render charts if attached
        if chart_imgs:
            for img_info in chart_imgs:
                if os.path.exists(img_info["path"]):
                    c1, c2 = st.columns([1.2, 1])
                    with c1:
                        st.image(img_info["path"], caption=f"📊 {img_info['title']}", use_container_width=True)
                    with c2:
                        st.caption(f"📁 **Filing:** `{img_info['filename']}` (Page {img_info['page']})")
                        st.caption(f"🎯 **BGE-Reranker Score:** `{img_info['score']}`")

        # 6. Render Verified SEC Sources
        if chunks:
            with st.expander(f"📋 Verified SEC Sources ({len(chunks)} Chunks)"):
                for idx, src in enumerate(chunks, 1):
                    st.caption(f"**Source {idx}:** `{src['filename']}` | Page {src['page']} | Modality: `{src['modality']}` | Score: `{src['rerank_score']}`")
                    st.markdown(src["content"][:300] + "...")
                    st.markdown("---")

        # 7. Evaluate Real-Time Metrics & Render Scorecard
        metrics_eval = FinancialRAGMetrics.evaluate_all(
            query=cleaned_query,
            answer=answer_text,
            context_chunks=chunks,
            start_time=start_time
        )

        is_fb = metrics_eval.get("is_fallback") or metrics_eval.get("faithfulness", {}).get("status") == "FALLBACK SAFE"
        scorecard_title = "🛡️ Out-of-Scope Fallback Scorecard (Safe Deflection - Zero Hallucination)" if is_fb else f"📊 Real-Time RAG Metrics & Audit Scorecard (Quality: {metrics_eval['quality_percentage']})"
        with st.expander(scorecard_title):
            mc1, mc2, mc3 = st.columns(3)
            with mc1:
                st.metric(label="🛡️ Faithfulness (Groundedness)", value=metrics_eval["faithfulness"]["percentage"])
                st.caption(f"Verified entities: {metrics_eval['faithfulness']['verified_count']} | Status: `{metrics_eval['faithfulness']['status']}`")
                st.metric(label="🎯 Relevance (Context & Answer)", value=metrics_eval["relevance"]["percentage"])
                st.caption(f"Context: {metrics_eval['relevance']['context_relevance']*100:.0f}% | Answer: {metrics_eval['relevance']['answer_relevance']*100:.0f}%")
            with mc2:
                st.metric(label="🔍 Retrieval Recall", value=metrics_eval["recall"]["percentage"])
                st.caption(f"Entities: {', '.join(metrics_eval['recall']['matched_entities'][:3]) or 'General query'}")
                st.metric(label="📝 Coherence Score", value=metrics_eval["coherence"].get("normalized_score", f"{metrics_eval['coherence'].get('score', 0.9):.2f} / 1.0"))
                st.caption(f"Scale: 0.0 – 1.0 ({metrics_eval['coherence'].get('percentage', '90.0%')}) | `{metrics_eval['coherence'].get('status', 'EXCELLENT')}`")
            with mc3:
                st.metric(label="📑 Citation Accuracy", value=metrics_eval["citation_accuracy"]["percentage"])
                st.caption(f"Valid: {metrics_eval['citation_accuracy']['valid_citations']}/{metrics_eval['citation_accuracy']['total_citations']} citations")
                st.metric(label="💰 Cost Per Query", value=metrics_eval["cost"]["formatted_cost"])
                st.caption(f"Tokens: {metrics_eval['cost']['total_tokens']} | Latency: {metrics_eval['latency_seconds']:.2f}s")

        # 8. Save completed response to messages history (ONLY active user and assistant messages)
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer_text,
            "sources": chunks,
            "chart_images": chart_imgs,
            "metrics": metrics_eval
        })

        # 9. Persist message and audit telemetry to PostgreSQL
        save_chat_message(
            role="assistant",
            content=answer_text,
            session_id=st.session_state.session_id,
            metadata={"quality": metrics_eval.get("quality_percentage"), "num_sources": len(chunks)}
        )
        log_query_audit(
            query=cleaned_query,
            answer=answer_text,
            metrics=metrics_eval,
            sources=chunks,
            session_id=st.session_state.session_id,
            is_fallback=is_fb
        )




# --- CHAT INPUT BAR (GEMINI STYLE) ---
user_query = st.chat_input("Ask Gemini about balance sheets, revenue numbers, or performance charts...")

if user_query:
    start_new_query(user_query)
    st.rerun()


# Disclaimer at bottom
st.markdown("""
<div class="gemini-disclaimer">
    Gemini may display inaccurate info, including about financial figures, so double-check verified SEC filings.
</div>
""", unsafe_allow_html=True)
