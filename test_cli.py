import os
import sys
import time
from pathlib import Path

# Ensure project root in sys.path
PROJECT_ROOT = str(Path(__file__).resolve().parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv()

from intent_classifier import FinancialIntentClassifier, get_conversational_greeting
from retriever import FinancialMultiModalRerankRetriever
from generator import UniversalFinancialGenerator
from rag_engine import FinancialGuardrails, FallbackManager
from metrics import FinancialRAGMetrics
from database import is_db_connected, log_query_audit

# Terminal ANSI Color Codes
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
MAGENTA = "\033[95m"
BOLD = "\033[1m"
RESET = "\033[0m"


def print_banner():
    print(f"\n{CYAN}{BOLD}{'='*75}")
    print("⚡ FINANCIAL MULTI-MODAL RAG — TERMINAL TESTING CONSOLE")
    print(f"{'='*75}{RESET}")
    print(f"PostgreSQL Database: {'🟢 Connected' if is_db_connected() else '🔴 Offline'}")
    print(f"Default LLM Provider: {BOLD}{os.getenv('LLM_PROVIDER', 'gemini').upper()}{RESET} ({os.getenv('LLM_MODEL', 'gemini-3.6-flash')})")
    print(f"Commands: Type your financial query, {YELLOW}'preset'{RESET} for sample questions, or {RED}'exit'{RESET} to quit.")
    print(f"{CYAN}{'-'*75}{RESET}\n")


PRESET_QUERIES = [
    "What was Apple's total iPhone net sales in Q2 2024 compared to Q2 2023?",
    "What were Amazon AWS net sales and growth rates in 2024?",
    "Show Google stock performance graph and cumulative returns",
    "Compare Apple and Google revenue 2024",
    "What was Tesla's net income in Q3 2024? (Tests Out of Scope)",
    "Hello, how can you help me today? (Tests Greeting)"
]


def run_single_query(query: str, retriever, generator, guardrails, fallback):
    start_time = time.time()
    print(f"\n{BOLD}❓ Query:{RESET} {query}")

    # 1. Guardrails Input Check
    input_check = guardrails.validate_input(query)
    if not input_check["passed"]:
        print(f"{RED}🛡️ Guardrail Blocked:{RESET} {input_check['reason']}")
        return

    # 2. Intent Classification
    intent = FinancialIntentClassifier.classify(query)
    intent_name = intent["intent"]
    conf_pct = int(intent["confidence"] * 100)
    entities = ", ".join(intent["companies"]) if intent.get("companies") else (intent.get("company") or "N/A")

    print(f"{MAGENTA}{BOLD}🎯 Intent:{RESET} {intent['badge_label']} ({intent_name}) | Confidence: {conf_pct}% | Entities: {entities}")

    # 3. Fast-Path Routing
    if intent_name == "GREETING_CHITCHAT":
        answer = get_conversational_greeting()
        chunks = []
    elif intent_name == "OUT_OF_SCOPE":
        answer = fallback.out_of_scope_fallback(query)
        chunks = []
    else:
        # Retrieve chunks
        print(f"{YELLOW}⏳ Retrieving and re-ranking SEC filings from Qdrant Cloud...{RESET}")
        retrieve_kwargs = {"query": query, "top_k": 3}
        if intent_name == "VISUAL_CHART_REQUEST":
            retrieve_kwargs["content_type"] = "chart"

        chunks = retriever.retrieve_and_rerank(**retrieve_kwargs)

        if not chunks:
            answer = fallback.out_of_scope_fallback(query)
        else:
            print(f"{GREEN}✓ Retrieved {len(chunks)} verified chunks (Top score: {chunks[0]['rerank_score']}){RESET}")
            print(f"{CYAN}⚡ Generating answer with {generator.DEFAULT_PROVIDER.upper()}...{RESET}")
            res = generator.generate(query, chunks)
            answer = res["answer"]

    latency = round(time.time() - start_time, 2)

    # 4. Display Answer
    print(f"\n{BOLD}{GREEN}📝 Verified Answer:{RESET}")
    print(f"{'-'*75}")
    print(answer)
    print(f"{'-'*75}")

    # 5. Display Sources
    if chunks:
        print(f"\n{BOLD}📋 Top Verified Sources:{RESET}")
        for idx, c in enumerate(chunks, 1):
            chart_info = f" | Chart: {c['image_path']}" if c.get("image_path") else ""
            print(f"  [{idx}] {c['company']} | {c['filename']} (Page {c['page']}) | Rerank Score: {c['rerank_score']}{chart_info}")

    # 6. Evaluation Metrics
    metrics_eval = FinancialRAGMetrics.evaluate_all(
        query=query,
        answer=answer,
        context_chunks=chunks,
        start_time=start_time
    )

    print(f"\n{YELLOW}{BOLD}📊 Real-Time Metrics Scorecard:{RESET}")
    print(f"  • Quality Score:      {BOLD}{metrics_eval.get('quality_percentage', 'N/A')}{RESET}")
    print(f"  • Faithfulness:       {metrics_eval['faithfulness']['percentage']} ({metrics_eval['faithfulness']['status']})")
    print(f"  • Context Relevance:  {metrics_eval['relevance']['context_relevance']*100:.0f}% | Answer: {metrics_eval['relevance']['answer_relevance']*100:.0f}%")
    print(f"  • Retrieval Recall:   {metrics_eval['recall']['percentage']}")
    print(f"  • Coherence Rating:   {metrics_eval['coherence'].get('normalized_score', '0.9/1.0')}")
    print(f"  • Cost & Latency:     {metrics_eval['cost']['formatted_cost']} | {latency}s")

    # 7. Audit Logging
    audit_id = log_query_audit(
        query=query,
        answer=answer,
        metrics=metrics_eval,
        sources=chunks,
        session_id="terminal-cli-tester",
        is_fallback=metrics_eval.get("is_fallback", False),
        intent=intent_name
    )
    if audit_id:
        print(f"  • PostgreSQL Log:     {GREEN}Saved as Audit ID #{audit_id}{RESET}")
    print(f"\n{CYAN}{'='*75}{RESET}\n")


def main():
    print_banner()

    print(f"{YELLOW}Initializing RAG Retriever & Generator...{RESET}")
    retriever = FinancialMultiModalRerankRetriever()
    generator = UniversalFinancialGenerator()
    guardrails = FinancialGuardrails()
    fallback = FallbackManager()
    print(f"{GREEN}✓ Ready for terminal testing!{RESET}\n")

    while True:
        try:
            user_input = input(f"{BOLD}Enter Query (or 'preset' / 'exit'): {RESET}").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit", "q"]:
                print(f"\n{GREEN}👋 Exiting Terminal Testing. Goodbye!{RESET}\n")
                break
            if user_input.lower() == "preset":
                print(f"\n{YELLOW}Sample Preset Questions:{RESET}")
                for idx, p in enumerate(PRESET_QUERIES, 1):
                    print(f"  [{idx}] {p}")
                choice = input(f"\n{BOLD}Select preset (1-{len(PRESET_QUERIES)}): {RESET}").strip()
                if choice.isdigit() and 1 <= int(choice) <= len(PRESET_QUERIES):
                    user_input = PRESET_QUERIES[int(choice) - 1]
                else:
                    print("Invalid choice, continuing...")
                    continue

            run_single_query(user_input, retriever, generator, guardrails, fallback)

        except KeyboardInterrupt:
            print(f"\n\n{GREEN}👋 Terminal Testing terminated.{RESET}\n")
            break
        except Exception as e:
            print(f"\n{RED}❌ Error running query: {e}{RESET}\n")


if __name__ == "__main__":
    main()
