from flask import Flask, request, jsonify
from flask_cors import CORS

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Settings,
    StorageContext,
    load_index_from_storage,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.llms import (
    LLM,
    ChatMessage,
    ChatResponse,
    CompletionResponse,
    LLMMetadata,
)

import requests
import json
import os
from typing import List, Optional, Iterator

# ======================================================
# LOAD API KEY
# ======================================================
def load_openrouter_key():
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY not set")
    return key


# ======================================================
# CONFIG
# ======================================================
DATA_DIR = "data"
PERSIST_DIR = "storage"

API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "openai/gpt-4o-mini"
OPENROUTER_API_KEY = load_openrouter_key()

# ======================================================
# FLASK APP
# ======================================================
app = Flask(__name__)
CORS(app)  # 🔥 FIXES CORS ISSUE

# ======================================================
# CUSTOM LLM (FULL IMPLEMENTATION)
# ======================================================
class OpenRouterLLM(LLM):
    model: str = MODEL
    api_key: str = OPENROUTER_API_KEY

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=4000,
            num_output=512,
            is_chat_model=True,
            model_name=self.model,
        )

    def _post(self, messages: List[ChatMessage]) -> dict:
        payload = {
            "model": self.model,
            "messages": [
                {"role": m.role, "content": m.content} for m in messages
            ],
            "max_tokens": 512,
        }

        resp = requests.post(
            API_URL,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            data=json.dumps(payload),
            timeout=60,
        )

        if resp.status_code != 200:
            raise RuntimeError(resp.text)

        return resp.json()

    # ---------- REQUIRED METHODS ----------
    def chat(self, messages, **kwargs) -> ChatResponse:
        data = self._post(messages)
        content = data["choices"][0]["message"]["content"]
        return ChatResponse(
            message=ChatMessage(role="assistant", content=content),
            raw=data,
        )

    def complete(self, prompt, **kwargs) -> CompletionResponse:
        data = self._post([ChatMessage(role="user", content=prompt)])
        return CompletionResponse(
            text=data["choices"][0]["message"]["content"],
            raw=data,
        )

    def stream_chat(self, messages, **kwargs):
        yield self.chat(messages)

    def stream_complete(self, prompt, **kwargs):
        yield self.complete(prompt)

    async def achat(self, messages, **kwargs):
        return self.chat(messages)

    async def acomplete(self, prompt, **kwargs):
        return self.complete(prompt)

    async def astream_chat(self, messages, **kwargs):
        yield self.chat(messages)

    async def astream_complete(self, prompt, **kwargs):
        yield self.complete(prompt)

# ======================================================
# LLAMA INDEX SETTINGS
# ======================================================
Settings.embed_model = HuggingFaceEmbedding(
    "sentence-transformers/all-MiniLM-L6-v2"
)
Settings.llm = OpenRouterLLM()

# ======================================================
# LOAD / BUILD INDEX
# ======================================================
_engine = None

def get_engine():
    global _engine
    if _engine:
        return _engine

    if os.path.exists(PERSIST_DIR):
        sc = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
        index = load_index_from_storage(sc)
    else:
        docs = SimpleDirectoryReader(DATA_DIR).load_data()
        index = VectorStoreIndex.from_documents(docs)
        index.storage_context.persist(persist_dir=PERSIST_DIR)

    _engine = index.as_query_engine(response_mode="compact")
    return _engine

# ======================================================
# API ROUTE
# ======================================================
@app.route("/api/chat", methods=["POST"])
def chat_api():
    data = request.get_json()
    question = data.get("question", "").strip()

    if not question:
        return jsonify({"error": "question is required"}), 400

    try:
        engine = get_engine()
        response = engine.query(question)
        answer = getattr(response, "response", str(response))
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ======================================================
# RUN SERVER
# ======================================================
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port)