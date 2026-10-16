import os
import time
from typing import List, Dict, Any
from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai.errors import ServerError, ClientError

load_dotenv()

SYSTEM_PROMPT = """You are an expert, precise, and strictly grounded academic assistant answering questions about the IIT Madras BS in Data Science & Applications program using official program documents.

Your instructions:
1. **Strict Grounding:** Answer ONLY using facts directly stated in the CONTEXT BLOCKS below. Never invent or assume facts not present in the text.
2. **Handle Specific Courses:** When the user asks about a specific course (such as DBMS, Python, Java, Machine Learning, PDSA, etc.), pay special attention to the blocks whose Section matches that course. If general program policies also apply, include them.
3. **Reproduce Exact Formulas:** When asked for a grading or score calculation formula, reproduce the EXACT formula as written in the document (including all weights, max terms, quiz components, and variable definitions such as GAA, Qz1, Qz2, OPPE, F, etc.).
4. **Preserve Dates & Eligibility:** Always include exact exam dates, shift timings (e.g., 09:30 AM to 11:30 AM), eligibility thresholds (e.g., >= 40/100, attendance rules), and syllabus coverage (e.g., Weeks 1 to 4).
5. **Inline Page Citations:** Cite every key fact, date, or formula with its source and page: for example `(grading_document.pdf, Page 32)`.
6. **No Speculation:** If the provided CONTEXT BLOCKS do not contain sufficient information to answer the question, state:
"I don't know based on the provided documents."
"""


class GroundedRAGEngine:
    def __init__(self, primary_model: str = "gemini-2.5-flash"):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("Missing GEMINI_API_KEY in .env file.")
        self.client = genai.Client(api_key=api_key)
        self.primary_model = primary_model

    def format_context(self, retrieved_chunks: List[Dict[str, Any]]) -> str:
        """Formats retrieved chunks with rich section, level, and page metadata headers."""
        formatted_blocks = []
        for idx, item in enumerate(retrieved_chunks, start=1):
            meta = item.get("metadata", {})
            source = meta.get("source", "Document")
            page = meta.get("page", "?")
            level = meta.get("level", "Document")
            section = meta.get("section", "General")
            rrf = item.get("rrf_score", "N/A")

            header = (
                f"--- CONTEXT BLOCK {idx} | Document: {source} (Page {page}) | "
                f"Level: {level} | Section: {section} | Match Rank Score: {rrf} ---"
            )
            block = f"{header}\n{item['text']}\n--- END BLOCK {idx} ---"
            formatted_blocks.append(block)

        return "\n\n".join(formatted_blocks)

    def answer_question(
        self,
        question: str,
        retrieved_chunks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generates a thoroughly grounded answer using hybrid-retrieved context."""
        if not retrieved_chunks:
            return {
                "answer": "I don't know based on the provided documents.",
                "sources": [],
                "retrieved_chunks": []
            }

        context_str = self.format_context(retrieved_chunks)

        user_prompt = (
            f"CONTEXT BLOCKS:\n\n{context_str}\n\n"
            f"========================================================\n"
            f"USER QUESTION: {question}\n"
            f"========================================================\n\n"
            f"Please provide a comprehensive, fully grounded answer based strictly on the CONTEXT BLOCKS above. "
            f"Include exact formulas, dates, criteria, and inline citations `(document, Page X)`. "
            f"If the answer is not present, reply: 'I don't know based on the provided documents.'"
        )

        # Robust API call with exponential backoff on transient errors (503 / 429)
        max_retries = 3
        last_error = None
        for attempt in range(max_retries + 1):
            try:
                response = self.client.models.generate_content(
                    model=self.primary_model,
                    contents=user_prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_PROMPT,
                        temperature=0.1,
                        max_output_tokens=4096
                    )
                )
                answer_text = response.text.strip()
                break
            except ServerError as e:
                last_error = e
                if attempt < max_retries:
                    wait_time = 2 ** (attempt + 1)
                    print(f"   ⏳ API busy (attempt {attempt + 1}/{max_retries}), retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    return {
                        "answer": f"⚠️ Gemini API unavailable after {max_retries} retries ({last_error}). Please try again in a moment.",
                        "sources": [],
                        "retrieved_chunks": retrieved_chunks
                    }

        # Deduplicate verified sources
        unique_sources = []
        for c in retrieved_chunks:
            meta = c.get("metadata", {})
            src_tag = f"{meta.get('source', 'doc')} (Page {meta.get('page', '?')})"
            if src_tag not in unique_sources:
                unique_sources.append(src_tag)

        if "i don't know" in answer_text.lower() and len(answer_text) < 60:
            unique_sources = []

        return {
            "answer": answer_text,
            "sources": unique_sources,
            "retrieved_chunks": retrieved_chunks
        }
