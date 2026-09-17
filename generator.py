import os
import base64
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from PIL import Image

load_dotenv()

# 1. Google Gemini Setup
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
try:
    from google import genai
    from google.genai import types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# 2. Mistral AI Setup
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
try:
    from mistralai.client import Mistral
    MISTRAL_AVAILABLE = True
except ImportError:
    MISTRAL_AVAILABLE = False

from rag_engine import FinancialGuardrails, FallbackManager


def encode_image_base64(image_path: str) -> str:
    """Encodes a local image file to base64 string for Mistral vision input."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


class UniversalFinancialGenerator:
    """
    Multi-Provider Generation Layer supporting:
    - 🔵 Google Gemini (gemini-3.6-flash / gemini-2.0-flash)
    - 🟠 Mistral AI (pixtral-12b-2409 / mistral-large-latest)
    
    Features:
    - Multi-Modal Input: Handles user query, financial markdown tables, and chart images.
    - Zero-Hallucination System Prompt: Strict adherence to SEC filing facts.
    - Groundedness Guardrail: Validates all numerical entities against retrieved chunks.
    - Graceful Fallback: Falls back to direct filing extract if API key is not configured or network fails.
    """

    DEFAULT_PROVIDER = os.getenv("LLM_PROVIDER", "gemini").lower()
    GEMINI_MODEL = os.getenv("LLM_MODEL", "gemini-3.6-flash")
    MISTRAL_MODEL = os.getenv("MISTRAL_MODEL", "pixtral-12b-2409")

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
        self.gemini_client = None
        self.mistral_client = None

        # Connect Google Gemini
        if GEMINI_API_KEY and GEMINI_AVAILABLE:
            try:
                gemini_url = os.getenv("GEMINI_API_URL")
                if gemini_url and "generativelanguage.googleapis.com" not in gemini_url:
                    self.gemini_client = genai.Client(api_key=GEMINI_API_KEY, http_options={"base_url": gemini_url})
                else:
                    self.gemini_client = genai.Client(api_key=GEMINI_API_KEY)
                print(f"✨ Connected to Google Gemini ({self.GEMINI_MODEL}) Generation Layer!")
            except Exception as e:
                print(f"⚠️ Failed to connect to Gemini API: {e}")

        # Connect Mistral AI
        mistral_key = os.getenv("MISTRAL_API_KEY")
        if mistral_key and MISTRAL_AVAILABLE:
            try:
                self.mistral_client = Mistral(api_key=mistral_key)
                print(f"🟠 Connected to Mistral AI ({self.MISTRAL_MODEL}) Generation Layer!")
            except Exception as e:
                print(f"⚠️ Failed to connect to Mistral API: {e}")
        elif not mistral_key:
            print("ℹ️ Note: MISTRAL_API_KEY not set in .env. Mistral ready on demand once key is provided.")

    def generate(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
        provider: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 1200
    ) -> Dict[str, Any]:
        """
        Generates a cited, verified financial answer using either Gemini or Mistral AI.
        """
        active_provider = (provider or self.DEFAULT_PROVIDER).lower()

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

        # Assemble Structured Context
        context_blocks = []
        for idx, chunk in enumerate(retrieved_chunks, 1):
            header = f"=== DOCUMENT {idx} [{chunk['modality']}] ==="
            source_meta = f"Company: {chunk['company']} | File: {chunk['filename']} | Page: {chunk['page']}"
            context_blocks.append(f"{header}\n{source_meta}\n\n{chunk['content']}")

        full_context_text = "\n\n".join(context_blocks)
        prompt_text = (
            f"### CONTEXT EXTRACTED FROM OFFICIAL SEC FILINGS:\n\n"
            f"{full_context_text}\n\n"
            f"### USER QUERY:\n{query}\n\n"
            f"### YOUR DETAILED & CITED FINANCIAL ANALYSIS:"
        )

        raw_answer = None
        used_model = None
        images_attached = 0

        # ==========================================
        # ROUTE 1: MISTRAL AI (PIXTRAL / LARGE)
        # ==========================================
        if active_provider == "mistral":
            if not self.mistral_client:
                # If Mistral key is not set, try falling back to Gemini if available
                if self.gemini_client:
                    print("⚠️ Mistral client not configured. Auto-routing to Google Gemini...")
                    active_provider = "gemini"
                else:
                    evidence = self.fallback.direct_evidence_fallback(query, retrieved_chunks)
                    return {
                        "answer": self.guardrails.validate_and_format_output(
                            "💡 **Mistral API Key Required:** Please add `MISTRAL_API_KEY` to your `.env` file to use Mistral AI.\n\n" + evidence
                        ),
                        "model": "mistral-key-missing-fallback",
                        "groundedness": {"passed": True, "score": 1.0},
                        "fallback_triggered": True,
                        "fallback_type": "MISTRAL_KEY_MISSING",
                        "sources": retrieved_chunks
                    }

            if active_provider == "mistral":
                try:
                    user_content = [{"type": "text", "text": prompt_text}]

                    # Attach images for Pixtral multi-modal vision
                    for chunk in retrieved_chunks:
                        img_path = chunk.get("image_path")
                        if img_path and os.path.exists(img_path):
                            try:
                                b64 = encode_image_base64(img_path)
                                user_content.append({
                                    "type": "image_url",
                                    "image_url": f"data:image/jpeg;base64,{b64}"
                                })
                                images_attached += 1
                            except Exception as e:
                                print(f"⚠️ Error encoding chart image for Mistral: {e}")

                    messages = [
                        {"role": "system", "content": self.SYSTEM_INSTRUCTION},
                        {"role": "user", "content": user_content if images_attached > 0 else prompt_text}
                    ]

                    response = self.mistral_client.chat.complete(
                        model=self.MISTRAL_MODEL,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens
                    )
                    raw_answer = response.choices[0].message.content
                    used_model = f"mistral/{self.MISTRAL_MODEL}"
                except Exception as e:
                    print(f"⚠️ Mistral API call failed: {e}")
                    # Try falling back to Gemini
                    if self.gemini_client:
                        print("Auto-falling back to Gemini...")
                        active_provider = "gemini"
                    else:
                        evidence = self.fallback.direct_evidence_fallback(query, retrieved_chunks)
                        return {
                            "answer": self.guardrails.validate_and_format_output(
                                f"⚠️ Mistral API Error ({e}). Falling back to direct filing extract:\n\n" + evidence
                            ),
                            "model": "mistral-api-fallback",
                            "groundedness": {"passed": True, "score": 1.0},
                            "fallback_triggered": True,
                            "fallback_type": "MISTRAL_API_ERROR",
                            "sources": retrieved_chunks
                        }

        # ==========================================
        # ROUTE 2: GOOGLE GEMINI
        # ==========================================
        if active_provider == "gemini":
            if not self.gemini_client:
                evidence = self.fallback.direct_evidence_fallback(query, retrieved_chunks)
                return {
                    "answer": self.guardrails.validate_and_format_output(evidence),
                    "model": "direct-evidence-fallback",
                    "groundedness": {"passed": True, "score": 1.0},
                    "fallback_triggered": True,
                    "fallback_type": "DIRECT_EVIDENCE_FALLBACK",
                    "sources": retrieved_chunks
                }

            try:
                contents = [prompt_text]
                attached_paths = set()
                for chunk in retrieved_chunks:
                    img_path = chunk.get("image_path")
                    if img_path and os.path.exists(img_path) and img_path not in attached_paths:
                        if images_attached >= 1:  # Attach top 1 primary visual chart to minimize payload & latency
                            break
                        try:
                            img = Image.open(img_path)
                            img.thumbnail((1024, 1024))  # Downscale to reduce network latency
                            contents.append(img)
                            attached_paths.add(img_path)
                            images_attached += 1
                        except Exception as e:
                            print(f"⚠️ Warning loading chart image ({img_path}): {e}")

                config = types.GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=min(max_tokens, 750),
                    system_instruction=self.SYSTEM_INSTRUCTION
                )

                response = self.gemini_client.models.generate_content(
                    model=self.GEMINI_MODEL,
                    contents=contents,
                    config=config
                )
                raw_answer = response.text
                used_model = f"gemini/{self.GEMINI_MODEL}"
            except Exception as e:
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
            rejection_text = (
                "🛡️ **Groundedness Guardrail Violation:**\n"
                f"The generated answer included unverified figures {groundedness_check['unverified_entities']} "
                "not backed by the retrieved filing tables.\n\n"
                + self.fallback.direct_evidence_fallback(query, retrieved_chunks)
            )
            return {
                "answer": self.guardrails.validate_and_format_output(rejection_text),
                "model": used_model,
                "groundedness": groundedness_check,
                "fallback_triggered": True,
                "fallback_type": "GROUNDEDNESS_GUARDRAIL_REJECTION",
                "sources": retrieved_chunks
            }

        # 4. Attach Compliance Disclaimer
        final_answer = self.guardrails.validate_and_format_output(raw_answer)

        return {
            "answer": final_answer,
            "model": used_model,
            "images_attached": images_attached,
            "groundedness": groundedness_check,
            "fallback_triggered": False,
            "fallback_type": None,
            "sources": retrieved_chunks
        }


# Backwards compatibility alias so existing scripts do not break
GeminiFinancialGenerator = UniversalFinancialGenerator
