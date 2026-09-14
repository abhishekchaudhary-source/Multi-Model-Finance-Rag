import time
from retriever import FinancialMultiModalRerankRetriever
from generator import GeminiFinancialGenerator
from metrics import FinancialRAGMetrics

def run_evaluation_benchmark():
    print("=" * 80)
    print("🧪 FINANCIAL MULTI-MODAL RAG BENCHMARK & METRICS EVALUATION")
    print("=" * 80)
    print("Evaluating 6 Production Metrics:")
    print("1. Relevance (Context & Answer)")
    print("2. Recall (Retrieval Coverage)")
    print("3. Faithfulness (Zero-Hallucination Groundedness)")
    print("4. Coherence Score (Structure & Clarity)")
    print("5. Citation Accuracy (Filing & Page Provenance)")
    print("6. Cost Per Query ($ Gemini 2.0 Flash Pricing)")
    print("=" * 80)

    retriever = FinancialMultiModalRerankRetriever()
    generator = GeminiFinancialGenerator()

    test_cases = [
        {
            "id": "TC-01",
            "category": "Table Extraction",
            "query": "What was Apple's total iPhone net sales in Q2 2024 compared to Q2 2023?",
            "ground_truth_entities": ["apple", "iphone", "2024", "2023", "q2", "sales"]
        },
        {
            "id": "TC-02",
            "category": "Segment Revenue",
            "query": "What were Amazon AWS net sales and growth rates in 2024?",
            "ground_truth_entities": ["amazon", "aws", "2024", "sales", "growth"]
        },
        {
            "id": "TC-03",
            "category": "Operating Expenses",
            "query": "What were Google's total research and development (R&D) expenses in 2023?",
            "ground_truth_entities": ["google", "2023", "r&d", "expenses"]
        },
        {
            "id": "TC-04",
            "category": "Multi-Modal Visual Chart",
            "query": "Show Google stock performance graph and cumulative returns",
            "ground_truth_entities": ["google", "stock", "performance", "graph", "returns"]
        },
        {
            "id": "TC-05",
            "category": "Negative / Out-of-Scope",
            "query": "What were Uber's gross bookings for Mobility vs Delivery in Q3 2024?",
            "ground_truth_entities": ["uber"]
        }
    ]

    total_faithfulness = 0.0
    total_relevance = 0.0
    total_recall = 0.0
    total_coherence = 0.0
    total_citation = 0.0
    total_cost = 0.0
    total_latency = 0.0

    scorecard_rows = []

    print("\n🚀 Running Benchmark Queries...\n")
    for tc in test_cases:
        print(f"🔹 [{tc['id']}] Category: {tc['category']}")
        print(f"   Query: \"{tc['query']}\"")
        
        start_time = time.time()
        
        # 1. Retrieve
        chunks = retriever.retrieve_and_rerank(
            query=tc["query"],
            initial_candidates=12,
            top_k=3,
            min_rerank_score=0.20
        )

        # 2. Generate
        if not chunks:
            gen_result = {
                "answer": (
                    "⚠️ **Out-of-Scope / Low Confidence Fallback:**\n\n"
                    f"The available SEC filings (Apple, Amazon, Google, Meta 2023–2025) do not contain verified data "
                    f"for query: *'{tc['query']}'*.\n"
                    "To prevent financial hallucinations, no unverified estimates or assumptions are provided."
                )
            }
        else:
            gen_result = generator.generate(tc["query"], chunks)

        # 3. Evaluate
        eval_report = FinancialRAGMetrics.evaluate_all(
            query=tc["query"],
            answer=gen_result["answer"],
            context_chunks=chunks,
            start_time=start_time,
            ground_truth_entities=tc.get("ground_truth_entities")
        )

        f_score = eval_report["faithfulness"]["score"]
        r_score = eval_report["relevance"]["score"]
        rc_score = eval_report["recall"]["score"]
        c_score = eval_report["coherence"]["score"]
        ca_score = eval_report["citation_accuracy"]["score"]
        cost_usd = eval_report["cost"]["cost_usd"]
        latency = eval_report["latency_seconds"]

        total_faithfulness += f_score
        total_relevance += r_score
        total_recall += rc_score
        total_coherence += c_score
        total_citation += ca_score
        total_cost += cost_usd
        total_latency += latency

        print(f"   ⚡ Overall Quality: {eval_report['quality_percentage']} | Faithfulness: {eval_report['faithfulness']['percentage']} | Latency: {latency:.2f}s | Cost: {eval_report['cost']['formatted_cost']}")
        print("-" * 80)

        scorecard_rows.append({
            "id": tc["id"],
            "category": tc["category"],
            "faithfulness": eval_report["faithfulness"]["percentage"],
            "relevance": eval_report["relevance"]["percentage"],
            "recall": eval_report["recall"]["percentage"],
            "coherence": eval_report["coherence"]["percentage"],
            "citation": eval_report["citation_accuracy"]["percentage"],
            "cost": eval_report["cost"]["formatted_cost"],
            "latency": f"{latency:.2f}s"
        })

    n = len(test_cases)
    avg_faithfulness = (total_faithfulness / n) * 100
    avg_relevance = (total_relevance / n) * 100
    avg_recall = (total_recall / n) * 100
    avg_coherence = (total_coherence / n) * 100
    avg_citation = (total_citation / n) * 100
    avg_cost = total_cost / n
    avg_latency = total_latency / n

    print("\n" + "=" * 92)
    print("🏆 FINAL FINANCIAL RAG EVALUATION SCORECARD MATRIX")
    print("=" * 92)
    header = f"{'ID':<6} | {'Category':<24} | {'Faithful':<9} | {'Relevance':<9} | {'Recall':<8} | {'Coherence':<9} | {'Citation':<8} | {'Cost ($)':<8} | {'Latency':<7}"
    print(header)
    print("-" * 92)
    for row in scorecard_rows:
        line = f"{row['id']:<6} | {row['category']:<24} | {row['faithfulness']:<9} | {row['relevance']:<9} | {row['recall']:<8} | {row['coherence']:<9} | {row['citation']:<8} | {row['cost']:<8} | {row['latency']:<7}"
        print(line)
    print("=" * 92)
    print("🎯 BENCHMARK SUMMARY AVERAGES:")
    print(f"   1. Faithfulness (Groundedness):   {avg_faithfulness:.1f}% (Target: 100%)")
    print(f"   2. Relevance (Context & Answer):  {avg_relevance:.1f}% (Target: >85%)")
    print(f"   3. Retrieval Recall:              {avg_recall:.1f}% (Target: >85%)")
    print(f"   4. Coherence Score:               {avg_coherence:.1f}% (Target: >80%)")
    print(f"   5. Citation Accuracy:             {avg_citation:.1f}% (Target: >95%)")
    print(f"   6. Average Cost Per Query:        ${avg_cost:.5f} (~0.02 INR)")
    print(f"   ⚡ Average Latency:                {avg_latency:.2f} seconds")
    print("=" * 92 + "\n")

if __name__ == "__main__":
    run_evaluation_benchmark()
