from __future__ import annotations

import requests


class LLMService:
    """
    Local LLM service using Ollama.

    No cloud API.
    No API key.
    Runs entirely on the local machine.
    """

    def __init__(
        self,
        model: str = "qwen3:4b",
        base_url: str = "http://127.0.0.1:11434",
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")

    def generate(
        self,
        question: str,
        context: str,
    ) -> str:

        if not question.strip():
            return (
                "I'm sorry, I didn't catch the question."
            )

        if not context.strip():
            return (
                "I'm sorry, I don't have enough "
                "information to answer that. "
                "Please contact reception."
            )

        system_prompt = """
You are the AI receptionist for
Rajalakshmi Engineering College (REC).

You answer questions using ONLY the
provided REC knowledge context.

STRICT RULES:

1. Use only information from the context.
2. Do not invent facts.
3. Do not use outside knowledge.
4. If the context does not contain enough
   information, say:
   "I'm sorry, I don't have enough information
   to answer that. Please contact reception."
5. Be concise and natural.
6. For comparison questions, compare the
   requested subjects clearly.
7. Preserve course codes, credits,
   objectives, units and course outcomes
   accurately.
8. Never mention the RAG system, embeddings,
   ChromaDB, retrieval or internal software.
9. You are speaking to a caller, so keep
   answers conversational.
"""

        user_prompt = f"""
KNOWLEDGE CONTEXT:

{context}

END KNOWLEDGE CONTEXT


USER QUESTION:

{question}


Answer the user's question using only the
knowledge context above.
"""

        payload = {
            "model": self.model,
            "system": system_prompt,
            "prompt": user_prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,
            },
        }

        try:

            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=120,
            )

            response.raise_for_status()

            data = response.json()

            answer = data.get(
                "response",
                "",
            ).strip()

            if not answer:
                return (
                    "I'm sorry, I wasn't able "
                    "to generate an answer."
                )

            return answer

        except requests.exceptions.ConnectionError:

            return (
                "I'm sorry, the local AI service "
                "is currently unavailable."
            )

        except requests.exceptions.Timeout:

            return (
                "I'm sorry, the AI service took "
                "too long to respond."
            )

        except requests.exceptions.RequestException as exc:

            print(
                f"[LLM ERROR] {exc}"
            )

            return (
                "I'm sorry, I couldn't process "
                "your request right now."
            )