"""
Production Financial Intent Classifier & Query Router.
Classifies user inquiries into deterministic execution intents:
1. GREETING_CHITCHAT: Direct fast conversational response (0ms vector latency).
2. OUT_OF_SCOPE: Immediate guardrail fallback for unsupported companies/topics.
3. VISUAL_CHART_REQUEST: Prioritizes chart/graph images and multi-modal evidence.
4. CROSS_COMPANY_COMPARISON: Multi-entity retrieval across Apple, Amazon, Google, Meta.
5. FINANCIAL_NUMERICAL_ANALYSIS: Standard deep hybrid retrieval + tabular grounding.
"""

import re
from typing import Dict, Any, List, Optional


class FinancialIntentClassifier:
    """
    High-performance, deterministic Intent Classifier and Entity Extractor
    tailored for SEC regulatory financial filings.
    """

    SUPPORTED_COMPANIES = ["Apple", "Amazon", "Google", "Meta"]
    
    UNINDEXED_COMPANIES = [
        "Tesla", "Microsoft", "Netflix", "Nvidia", "Uber", "Airbnb", 
        "Reliance", "Intel", "Salesforce", "Twitter", "Boeing", "IBM", 
        "Spotify", "Oracle", "Walmart", "Disney", "OpenAI", "TCS", "Infosys"
    ]

    GREETING_PATTERNS = [
        r"^(hi|hello|hey|greetings|good\s+(morning|afternoon|evening))\b",
        r"^who\s+are\s+you\b",
        r"^what\s+(can\s+you\s+do|are\s+your\s+capabilities)\b",
        r"^(help|how\s+does\s+this\s+work)\b"
    ]

    CHART_PATTERNS = [
        r"\b(chart|graph|plot|diagram|visual|trendline|cumulative\s+total\s+return|stock\s+performance)\b"
    ]

    COMPARISON_PATTERNS = [
        r"\b(compare|versus|vs\.?|comparison|better|higher|lower\s+than|difference\s+between)\b"
    ]

    INTENTS = {
        "GREETING_CHITCHAT": "Conversational inquiry / greeting requiring immediate response without vector search.",
        "OUT_OF_SCOPE": "Inquiry regarding unsupported entities or non-financial domains.",
        "VISUAL_CHART_REQUEST": "User explicitly requesting visual chart/graph evidence from filings.",
        "CROSS_COMPANY_COMPARISON": "Comparative query spanning multiple corporate filings.",
        "FINANCIAL_NUMERICAL_ANALYSIS": "Standard financial metrics extraction (revenue, net income, margins, R&D)."
    }

    @classmethod
    def classify(cls, user_query: str) -> Dict[str, Any]:
        """
        Classifies user query and extracts relevant financial metadata.
        Returns:
            Dict containing intent, confidence, detected companies, year, quarter, and modality.
        """
        query_clean = user_query.strip()
        query_lower = query_clean.lower()

        # 1. Detect Supported Companies
        detected_companies = []
        for comp in cls.SUPPORTED_COMPANIES:
            if re.search(r'\b' + comp + r'\b', query_clean, re.IGNORECASE):
                detected_companies.append(comp)

        # 2. Detect Unindexed Companies
        detected_unindexed = []
        for un_comp in cls.UNINDEXED_COMPANIES:
            if re.search(r'\b' + un_comp + r'\b', query_clean, re.IGNORECASE):
                detected_unindexed.append(un_comp)

        # 3. Detect Year
        year_match = re.search(r'\b(202[0-6])\b', query_clean)
        year = year_match.group(1) if year_match else None

        # 4. Detect Quarter
        quarter_match = re.search(r'\b(q[1-4])\b', query_clean, re.IGNORECASE)
        quarter = quarter_match.group(1).upper() if quarter_match else None

        # 5. Check Greeting / Chitchat Intent
        for pattern in cls.GREETING_PATTERNS:
            if re.search(pattern, query_lower):
                return {
                    "intent": "GREETING_CHITCHAT",
                    "confidence": 0.98,
                    "company": None,
                    "companies": [],
                    "year": None,
                    "quarter": None,
                    "content_type": None,
                    "badge_label": "👋 Greeting / About",
                    "badge_color": "#1A73E8",
                    "action": "DIRECT_CONVERSATIONAL_RESPONSE",
                    "explanation": "Query is a greeting or general assistance inquiry."
                }

        # 6. Check Out-of-Scope Intent (Explicit Unindexed Company or non-financial domain)
        if detected_unindexed and not detected_companies:
            return {
                "intent": "OUT_OF_SCOPE",
                "confidence": 0.99,
                "company": detected_unindexed[0],
                "companies": detected_unindexed,
                "unindexed_company": detected_unindexed[0],
                "year": year,
                "quarter": quarter,
                "content_type": None,
                "badge_label": "🛡️ Out of Scope",
                "badge_color": "#EA4335",
                "action": "IMMEDIATE_FALLBACK_DISCLAIMER",
                "explanation": f"Inquiry targets unsupported company: '{detected_unindexed[0]}'."
            }

        # 7. Check Cross-Company Comparison Intent
        is_comparison_keyword = any(re.search(p, query_lower) for p in cls.COMPARISON_PATTERNS)
        if len(detected_companies) >= 2 or (is_comparison_keyword and len(detected_companies) >= 1):
            return {
                "intent": "CROSS_COMPANY_COMPARISON",
                "confidence": 0.95,
                "company": detected_companies[0] if detected_companies else None,
                "companies": detected_companies,
                "year": year,
                "quarter": quarter,
                "content_type": "table",
                "badge_label": "🔀 Multi-Company Comparison",
                "badge_color": "#FBBC04",
                "action": "MULTI_ENTITY_RETRIEVAL",
                "explanation": f"Comparative analysis requested across {detected_companies}."
            }

        # 8. Check Visual Chart Request Intent
        is_chart_request = any(re.search(p, query_lower) for p in cls.CHART_PATTERNS)
        if is_chart_request:
            return {
                "intent": "VISUAL_CHART_REQUEST",
                "confidence": 0.96,
                "company": detected_companies[0] if detected_companies else None,
                "companies": detected_companies,
                "year": year,
                "quarter": quarter,
                "content_type": "chart",
                "badge_label": "🖼️ Visual Chart Request",
                "badge_color": "#A142F4",
                "action": "CHART_IMAGE_PRIORITY_RETRIEVAL",
                "explanation": "User explicitly requested visual graphs or charts."
            }

        # 9. Default: Financial Numerical & Text Analysis
        return {
            "intent": "FINANCIAL_NUMERICAL_ANALYSIS",
            "confidence": 0.92,
            "company": detected_companies[0] if detected_companies else None,
            "companies": detected_companies,
            "year": year,
            "quarter": quarter,
            "content_type": "table" if any(t in query_lower for t in ["table", "balance sheet", "income statement", "revenue by"]) else None,
            "badge_label": "📊 Financial Numerical Analysis",
            "badge_color": "#34A853",
            "action": "HYBRID_RETRIEVAL_AND_RERANKING",
            "explanation": "Standard financial filing metrics & statement inquiry."
        }


def get_conversational_greeting() -> str:
    """Provides a warm, expert financial analyst greeting for chitchat queries."""
    return (
        "👋 **Hello! I am your AI Financial Intelligence Analyst.**\n\n"
        "I am specialized in deep regulatory analysis of official **SEC Filings (Form 10-K, 10-Q, 8-K)** "
        "and visual stock performance charts for:\n"
        "- 🍏 **Apple Inc.** (AAPL)\n"
        "- 📦 **Amazon.com, Inc.** (AMZN)\n"
        "- 🔍 **Alphabet Inc. / Google** (GOOGL)\n"
        "- 👥 **Meta Platforms, Inc.** (META)\n\n"
        "**You can ask me to:**\n"
        "1. *Extract exact financial numbers* (e.g., *'What was Apple's net sales in Q2 2024?'*)\n"
        "2. *Display visual stock charts* (e.g., *'Display Meta stock performance and cumulative return visual chart'*)\n"
        "3. *Analyze segment revenues* (e.g., *'What were Amazon's AWS revenues in 2024?'*)\n"
        "4. *Compare company metrics* (e.g., *'Compare Google and Apple R&D expenses in 2023'*)\n\n"
        "How can I assist your financial research today?"
    )
