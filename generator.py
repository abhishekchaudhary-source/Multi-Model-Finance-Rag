import os
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from PIL import Image

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("LLM_API_KEY")

try:
    from google import genai
    from google.genai import types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

from rag_engine import FinancialGuardrails, FallbackManager


class GeminiFinancialGenerator:
    """
    Generation Layer powered by Google Gemini 2.0 Flash:
    - Multi-Modal Input: Simultaneously accepts user queries, Markdown financial tables, and chart images.
    - Zero-Hallucination System Prompt: Enforces strict adherence to filing facts and numerical precision.
    - Fact-Checking Guardrail: Runs Groundedness validation on the generated answer.
    - Graceful Fallback: Seamlessly falls back to direct filing evidence if API key is not set or network fails.
    """

    MODEL_NAME = os.getenv("LLM_MODEL", "gemini-3.6-flash")

    SYSTEM_INSTRUCTION = """You are a senior financial analyst specialized in SEC regulatory filings (10-K, 10-Q, 8-K) for Apple, Amazon, Google, and Meta.

STRICT OPERATING PRINCIPLES:
1. Grounding: Answer strictly and exclusively using the provided context (tables, charts, narrative text).
2. Numerical Precision: Never fabricate, assume, or extrapolate figures. Always preserve exact currency ($), units (millions/billions), and percentages (%).
3. Source Attribution: Explicitly cite the document name, year/quarter, and page number for every key metric or statement (e.g., [Source: Apple 10-Q Q2 2024, Page 18]).
4. Visual Reasoning: If a financial chart or graph image is attached, describe the visual trendlines, axis metrics, and shareholder return accurately.
5. Incomplete Data: If the context does not contain the answer, explicitly state: "The provided SEC filings do not contain verified data for this specific metric." Do not guess.
"""

    def __init__(self):
        self.guardrails = FinancialGuardrails()
        self.fallback = FallbackManager()
        self.client = None

        gemini_url = os.getenv("GEMINI_API_URL") or os.getenv("LLM_API_URL")
        if GEMINI_API_KEY and GEMINI_AVAILABLE:
            try:
                if gemini_url and "generativelanguage.googleapis.com" not in gemini_url:
                    self.client = genai.Client(api_key=GEMINI_API_KEY, http_options={"base_url": gemini_url})
                else:
                    self.client = genai.Client(api_key=GEMINI_API_KEY)
                print(f"✨ Connected to Google Gemini ({self.MODEL_NAME}) Generation Layer!")
            except Exception as e:
                print(f"⚠️ Failed to connect to Gemini API: {e}")
        else:
            print("ℹ️ Note: GEMINI_API_KEY not found in .env. Running in Direct-Evidence Fallback Mode.")

    def generate(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
        temperature: float = 0.0,
        max_tokens: int = 1200
    ) -> Dict[str, Any]:
        """
        Generates a cited, verified financial answer using Gemini 2.0 Flash.
        """
        if not retrieved_chunks:
            fallback_text = self.fallback.out_of_scope_fallback(query)
            return {
                "answer": self.guardrails.validate_and_format_output(fallback_text),
                "model": "fallback",
                "groundedness": {"passed": True, "score": 1.0},
                "fallback_triggered": True,
                "fallback_type": "NO_CONTEXT_RETRIEVED"
            }

        source_texts = []
        for c in retrieved_chunks:
            meta_snippet = f"{c.get('company', '')} {c.get('filename', '')} page {c.get('page', '')} {c.get('year', '')} {c.get('quarter', '')} {c.get('doc_type', '')}"
            source_texts.append(f"{meta_snippet} {c.get('content', '')}")

        # Check if Gemini API is available
        if not self.client:
            evidence = self.fallback.direct_evidence_fallback(query, retrieved_chunks)
            return {
                "answer": self.guardrails.validate_and_format_output(evidence),
                "model": "direct-evidence-fallback",
                "groundedness": {"passed": True, "score": 1.0, "status": "Direct SEC Evidence"},
                "fallback_triggered": True,
                "fallback_type": "DIRECT_EVIDENCE_FALLBACK",
                "sources": retrieved_chunks
            }

        # 1. Assemble Context for Gemini
        context_blocks = []
        for idx, chunk in enumerate(retrieved_chunks, 1):
            header = f"=== DOCUMENT {idx} [{chunk['modality']}] ==="
            source_meta = f"Company: {chunk['company']} | File: {chunk['filename']} | Page: {chunk['page']}"
            context_blocks.append(f"{header}\n{source_meta}\n\n{chunk['content']}")

        full_context_text = "\n\n".join(context_blocks)

        prompt = (
            f"### CONTEXT EXTRACTED FROM OFFICIAL SEC FILINGS:\n\n"
            f"{full_context_text}\n\n"
            f"### USER QUERY:\n{query}\n\n"
            f"### YOUR DETAILED & CITED FINANCIAL ANALYSIS:"
        )

        # Prepare Multi-Modal Contents list
        contents = [prompt]

        # Attach chart images if present in top results
        images_attached = 0
        for chunk in retrieved_chunks:
            img_path = chunk.get("image_path")
            if img_path and os.path.exists(img_path):
                try:
                    img = Image.open(img_path)
                    contents.append(img)
                    images_attached += 1
                except Exception as e:
                    print(f"⚠️ Warning loading chart image ({img_path}): {e}")

        # 2. Call Gemini 2.0 Flash
        try:
            config = types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
                system_instruction=self.SYSTEM_INSTRUCTION
            )

            response = self.client.models.generate_content(
                model=self.MODEL_NAME,
                contents=contents,
                config=config
            )
            raw_answer = response.text
        except Exception as e:
            # Fallback Tier 3: API Failure Fallback
            evidence = self.fallback.direct_evidence_fallback(query, retrieved_chunks)
            err_msg = f"⚠️ Gemini API Call Failed ({e}). Falling back to direct filing extract:\n\n" + evidence
            return {
                "answer": self.guardrails.validate_and_format_output(err_msg),
                "model": "api-error-fallback",
                "groundedness": {"passed": True, "score": 1.0},
                "fallback_triggered": True,
                "fallback_type": "API_ERROR_FALLBACK",
                "sources": retrieved_chunks
            }

        # 3. Groundedness Evaluation Guardrail
        groundedness_check = self.guardrails.validate_groundedness(raw_answer, source_texts, query=query)

        if not groundedness_check["passed"]:
            # Hallucination detected: Reject answer and fallback
            rejection_text = (
                "🛡️ **Groundedness Guardrail Violation:**\n"
                f"The generated answer included unverified figures {groundedness_check['unverified_entities']} "
                "not backed by the retrieved filing tables.\n\n"
                + self.fallback.direct_evidence_fallback(query, retrieved_chunks)
            )
            return {
                "answer": self.guardrails.validate_and_format_output(rejection_text),
                "model": self.MODEL_NAME,
                "groundedness": groundedness_check,
                "fallback_triggered": True,
                "fallback_type": "GROUNDEDNESS_GUARDRAIL_REJECTION",
                "sources": retrieved_chunks
            }

        # 4. Attach Compliance Disclaimer
        final_answer = self.guardrails.validate_and_format_output(raw_answer)

        return {
            "answer": final_answer,
            "model": self.MODEL_NAME,
            "images_attached": images_attached,
            "groundedness": groundedness_check,
            "fallback_triggered": False,
            "fallback_type": None,
            "sources": retrieved_chunks
        }


def run_interactive_generator():
    from retriever import FinancialMultiModalRerankRetriever

    retriever = FinancialMultiModalRerankRetriever()
    generator = GeminiFinancialGenerator()

    print("\n" + "=" * 70)
    print("🤖 Financial Multi-Modal RAG (Google Gemini 2.0 Flash + BGE-Reranker-v2-m3)")
    print("Type your financial query below (or 'exit' to quit).")
    print("=" * 70 + "\n")

    while True:
        try:
            query = input("❓ Ask a Financial Question: ").strip()
            if not query:
                continue
            if query.lower() in ["exit", "quit"]:
                print("👋 Exiting generator. Goodbye!")
                break

            # 1. Input Guardrail
            input_check = generator.guardrails.validate_input(query)
            if not input_check["passed"]:
                print(f"\n🛡️ Guardrail Block: {input_check['reason']}\n")
                continue

            print("\n⏳ Retrieving and Re-ranking verified SEC data...")
            chunks = retriever.retrieve_and_rerank(query=query, top_k=3)

            print("⚡ Generating answer with Google Gemini 2.0 Flash...\n")
            result = generator.generate(query=query, retrieved_chunks=chunks)

            print("=" * 70)
            print(result["answer"])
            print("=" * 70)
            print(f"📊 Model: {result['model']} | Groundedness: {result['groundedness']['groundedness_score']} | Fallback: {result['fallback_triggered']}\n")

        except KeyboardInterrupt:
            print("\n👋 Terminated.")
            break
        except Exception as e:
            print(f"❌ Error: {e}\n")


if __name__ == "__main__":
    run_interactive_generator()
