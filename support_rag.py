import hashlib
import json
import logging
import os
import re
import unicodedata
from collections import Counter
from typing import List, TypedDict

from dotenv import load_dotenv

load_dotenv()  

import pymupdf as fitz  
import requests
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.documents import Document
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, START, StateGraph

log = logging.getLogger("support_rag")

# Configuration 
SOURCE_PDF = os.getenv("SOURCE_PDF", "data/Expt6_Source_File.pdf")
INDEX_DIR = os.getenv("INDEX_DIR", "faiss_index")
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
TOP_K = int(os.getenv("TOP_K", "3"))
MIN_RELEVANCE = float(os.getenv("MIN_RELEVANCE", "0.40"))
LLM_BACKEND = os.getenv("LLM_BACKEND", "ollama").strip().lower()
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "60"))
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "200"))


# 1. LLM backends 
class BackendError(Exception):
    """The LLM backend could not produce an answer.
    The message is for server logs only - it is never sent to the client."""


class OllamaBackend:
    """Option B - local model served by Ollama (default http://localhost:11434)."""
    name = "ollama"

    def __init__(self):
        self.base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
        self.model = os.getenv("OLLAMA_MODEL", "qwen2.5:0.5b")

    def generate(self, messages: List[dict]) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": 0, "num_predict": MAX_NEW_TOKENS, "repeat_penalty": 1.1},
        }
        try:
            r = requests.post(f"{self.base_url}/api/chat", json=payload, timeout=LLM_TIMEOUT)
        except requests.RequestException as e:
            raise BackendError(f"Ollama unreachable at {self.base_url} ({type(e).__name__})") from e
        if r.status_code == 404:
            raise BackendError(f"Ollama model '{self.model}' not found - run: ollama pull {self.model}")
        if r.status_code != 200:
            raise BackendError(f"Ollama returned HTTP {r.status_code}")
        try:
            return r.json()["message"]["content"].strip()
        except (KeyError, ValueError) as e:
            raise BackendError("Ollama returned an unexpected response format") from e

    def ping(self) -> bool:
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=3)
            names = {m.get("name", "") for m in r.json().get("models", [])}
            return r.ok and (self.model in names or f"{self.model}:latest" in names)
        except Exception:
            return False


class CloudBackend:
    """Option A - hosted, OpenAI-compatible chat-completions API
    (e.g. Hugging Face Inference Providers router, Groq, OpenRouter ...)."""
    name = "cloud"

    def __init__(self):
        self.base_url = os.getenv("CLOUD_BASE_URL", "https://router.huggingface.co/v1").rstrip("/")
        self.model = os.getenv("CLOUD_MODEL", "").strip()
        self._api_key = os.getenv("CLOUD_API_KEY", "").strip()  # read from env, never hard-coded
        if not self.model or not self._api_key:
            raise RuntimeError("LLM_BACKEND=cloud needs CLOUD_MODEL and CLOUD_API_KEY in .env")

    def generate(self, messages: List[dict]) -> str:
        payload = {"model": self.model, "messages": messages,
                   "max_tokens": MAX_NEW_TOKENS, "temperature": 0}
        try:
            r = requests.post(f"{self.base_url}/chat/completions", json=payload, timeout=LLM_TIMEOUT,
                              headers={"Authorization": f"Bearer {self._api_key}"})
        except requests.RequestException as e:
            raise BackendError(f"Cloud API unreachable ({type(e).__name__})") from e
        if r.status_code in (401, 403):
            raise BackendError(f"Cloud API rejected the key (HTTP {r.status_code}) - check CLOUD_API_KEY")
        if r.status_code == 402:
            raise BackendError("Cloud API free credits used up (HTTP 402) - switch provider or add credits")
        if r.status_code == 429:
            raise BackendError("Cloud API quota / rate limit exceeded (HTTP 429)")
        if r.status_code != 200:
            raise BackendError(f"Cloud API returned HTTP {r.status_code} - check CLOUD_MODEL / CLOUD_BASE_URL"
                               f" | {r.text[:200]}")   # provider's message (e.g. model not found); logs only
        try:
            return r.json()["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, ValueError, TypeError) as e:
            raise BackendError("Cloud API returned an unexpected response format") from e

    def ping(self) -> bool:
        try:
            r = requests.get(f"{self.base_url}/models", timeout=5,
                             headers={"Authorization": f"Bearer {self._api_key}"})
            return r.ok
        except Exception:
            return False


class MockBackend:
    """No LLM at all - returns the top retrieved passage. Use it to test the API
    structure when neither Ollama nor a cloud quota is available."""
    name = "mock"

    def generate(self, messages: List[dict]) -> str:
        user = messages[-1]["content"]
        m = re.search(r"\[1\] \[[^\]]*\] (.+?)(?:\n\n\[2\]|\n\nCUSTOMER QUESTION)", user, re.S)
        return (m.group(1).strip()[:400] if m else "INSUFFICIENT_CONTEXT")

    def ping(self) -> bool:
        return True


def make_backend():
    if LLM_BACKEND == "ollama":
        return OllamaBackend()
    if LLM_BACKEND == "cloud":
        return CloudBackend()
    if LLM_BACKEND == "mock":
        return MockBackend()
    raise RuntimeError(f"Unknown LLM_BACKEND '{LLM_BACKEND}' (use ollama | cloud | mock)")


# 2. Knowledge base: PDF -> sections -> chunks   (Expt 6, Steps 3-5)
BULLET_RE = re.compile(r"^(?:[•●▪○◦‣■]|[-–—])\s*")
NUMBERED_RE = re.compile(r"^\d+[.)]\s*")


def normalize_line(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    for ch in ("​", "﻿", "­"):
        text = text.replace(ch, "")
    return re.sub(r"[ \t]+", " ", text).strip()


def read_pdf_lines(path: str):
    """Return [(font_size, text, page_no), ...] in reading order."""
    out = []
    with fitz.open(path) as doc:
        for page_no, page in enumerate(doc, 1):
            for block in page.get_text("dict")["blocks"]:
                if block.get("type") != 0:
                    continue
                for line in block["lines"]:
                    spans = [s for s in line["spans"] if s["text"].strip()]
                    if not spans:
                        continue
                    text = normalize_line("".join(s["text"] for s in line["spans"]))
                    if text:
                        out.append((round(max(s["size"] for s in spans), 1), text, page_no))
    return out


def join_lines(lines: List[str]) -> str:
    """Re-join wrapped PDF lines; keep list items and finished sentences on their own line."""
    out, pending_bullet = [], False
    for line in lines:
        if BULLET_RE.fullmatch(line):          # bullet glyph on its own line
            pending_bullet = True
            continue
        is_item = pending_bullet or BULLET_RE.match(line) or NUMBERED_RE.match(line)
        if BULLET_RE.match(line):
            line = BULLET_RE.sub("", line)
        if is_item and not NUMBERED_RE.match(line):
            line = "- " + line
        pending_bullet = False
        if not out or is_item or re.search(r'[.:!?"”]$', out[-1]):
            out.append(line)
        else:
            out[-1] += " " + line
    return "\n".join(out)


def build_sections(lines):
    """Large-font lines are article titles; everything below a title belongs to that section."""
    body_size = Counter(size for size, _, _ in lines).most_common(1)[0][0]
    ratio = 1.25     
    if sum(1 for s, _, _ in lines if s >= body_size * ratio) < 10:  
        ratio = 1.1
    title_min = body_size * ratio

    sections, title, buf, page, prev_title = [], "General", [], 1, False

    def flush():
        if buf:
            sections.append({"section": title, "page": page, "text": join_lines(buf)})
            buf.clear()

    for size, text, page_no in lines:
        if size >= title_min:
            if prev_title and not buf:      
                title = f"{title} {text}"
            else:
                flush()
                title = text
            prev_title = True
            continue
        prev_title = False
        if not buf:
            page = page_no
        buf.append(text)
    flush()
    return sections


def build_chunks(pdf_path: str) -> List[Document]:
    sections = build_sections(read_pdf_lines(pdf_path))
    source = os.path.basename(pdf_path)
    docs = [Document(page_content=s["text"],
                     metadata={"source": source, "section": s["section"], "page": s["page"]})
            for s in sections if len(s["text"]) > 30]
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=80,
                                              separators=["\n", ". ", " ", ""])
    chunks = splitter.split_documents(docs)
    for c in chunks:  
        c.page_content = f"[{c.metadata['section']}] {c.page_content}"
    log.info("knowledge base: %d sections -> %d chunks", len(docs), len(chunks))
    return chunks


def load_or_build_index(embeddings):
    """Build the FAISS index once and cache it on disk; rebuild only when the PDF or model changes."""
    # Content hash, not path/mtime: Docker layers can change both, which forced a rebuild at startup
    with open(SOURCE_PDF, "rb") as f:
        pdf_sha256 = hashlib.sha256(f.read()).hexdigest()
    fingerprint = {"pdf_sha256": pdf_sha256, "embed_model": EMBED_MODEL, "embedder": "fastembed"}
    meta_path = os.path.join(INDEX_DIR, "meta.json")
    if os.path.exists(meta_path):
        try:
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
            if meta.get("fingerprint") == fingerprint:
                store = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True,
                                         distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT)
                log.info("loaded cached FAISS index from %s", INDEX_DIR)
                return store, meta["n_chunks"]
        except Exception:
            log.warning("cached index unreadable - rebuilding")

    chunks = build_chunks(SOURCE_PDF)
    store = FAISS.from_documents(chunks, embeddings, distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT)
    store.save_local(INDEX_DIR)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({"fingerprint": fingerprint, "n_chunks": len(chunks)}, f)
    log.info("built and saved FAISS index (%d chunks) to %s", len(chunks), INDEX_DIR)
    return store, len(chunks)


# 3. Workflow pieces   (Expt 6, Steps 8-12)
class SupportState(TypedDict, total=False):
    question: str
    intent: str
    search_query: str
    context: str
    sources: List[str]
    top_score: float
    draft: str
    review_note: str
    retry_count: int
    final_answer: str
    used_fallback: bool


INTENT_RULES = [
    ("order_tracking",         ["track", "shipment", "dispatch", "delivery", "order status", "status of order",
                                "live status", "order reference", "where is my order"]),
    ("account_access",         ["password", "sign in", "sign-in", "login", "log in", "verification code",
                                "one-time code", "reset link"]),
    ("billing_payments",       ["charged", "billing", "invoice", "card number", "payment status", "transaction",
                                "update payment", "payment card"]),
    ("technical_support",      ["website", "not loading", "browser", "cache", "error message"]),
    ("kindle_gifts",           ["gift", "redeem", "redemption", "buy for others", "recipient", "prepaid", "resell"]),
    ("kindle_returns_refunds", ["return a kindle", "kindle book for a refund", "kindle refund", "return the book",
                                "return a book", "partially read"]),
    ("returns_refunds",        ["refund", "return", "exchange", "packaging"]),
    ("send_to_kindle",         ["send to kindle", "send documents", "document", "file type", "file size",
                                "approved email", "archiv", "attachment"]),
    ("kindle_subscriptions",   ["kindle unlimited", "prime reading", "borrow", "subscription", "pause"]),
    ("prime_membership",       ["prime", "twitch", "auto-renew", "renew", "annual plan", "monthly plan"]),
    ("fire_tv",                ["fire tv", "remote", "hdmi", "parental", "offload", "home theatre", "home theater"]),
    ("echo_accessibility",     ["echo", "alexa", "voiceview", "caption", "magnifier", "adaptive listening",
                                "tap to alexa", "show and tell"]),
    ("gaming_luna",            ["luna", "game", "vsync", "claim", "my collection"]),
    ("kindle_reading",         ["kindle", "sync", "word wise", "book version", "transfer", "read aloud", "accessibility"]),
    ("contact_support",        ["contact", "support team", "help section"]),
]

CATEGORY_TERMS = {
    "order_tracking": "order dispatch tracking shipment status Orders section",
    "returns_refunds": "return eligibility refund process original payment method",
    "account_access": "account sign in Forgot Password reset never share password",
    "billing_payments": "billing payment charged twice transaction date order reference",
    "technical_support": "website not loading browser cache troubleshooting",
    "contact_support": "contact support team Help section",
    "kindle_returns_refunds": "return Kindle book for refund Manage Your Content and Devices",
    "kindle_gifts": "Kindle book gift redeem redemption link unredeemed refund recipient",
    "send_to_kindle": "Send to Kindle email documents supported file types approved email list",
    "prime_membership": "Prime membership end membership plan auto-renew benefits",
    "kindle_subscriptions": "Kindle Unlimited Prime Reading borrow titles cancel pause",
    "kindle_reading": "Kindle book sync library update version country transfer",
    "fire_tv": "Fire TV settings remote software update HDMI parental controls",
    "echo_accessibility": "Echo Show Settings Accessibility captioning VoiceView",
    "gaming_luna": "Amazon Luna games claim Amazon Games app troubleshooting",
    "general_support": "customer support help contact",
}

SYSTEM_PROMPT = """You are a polite customer-support assistant.
Use only the FAQ CONTEXT to answer the CUSTOMER QUESTION.
Do not invent order status, refund approvals, timelines, phone numbers, or company policies.
If the FAQ says the assistant cannot do something (for example check live order status or reveal a password), say so and give the FAQ's alternative.
If the context does not contain the answer, output exactly:
INSUFFICIENT_CONTEXT
Do not ask for passwords, one-time codes, or full payment-card details.
Keep the answer clear and concise: at most 4 sentences or a short numbered list."""

FALLBACK_TEXT = (
    "I couldn't find a reliable answer in the available FAQ. "
    "Please contact the support team through the Help section. "
    "Do not share your password, one-time code, or full payment-card details."
)
CLARIFY_TEXT = (
    "Could you tell me a little more? Please mention which product or service you mean "
    "(for example Kindle, Fire TV, Echo, Prime, or an order) and what is happening, so I can find the right FAQ answer. "
    "Do not share your password, one-time code, or full payment-card details."
)

VAGUE_WORDS = {"it", "this", "that", "they", "them", "its"}
LIST_MARKER_RE = re.compile(r"(?m)^\s*\d+[.)]\s+")


def is_vague(question: str, intent: str) -> bool:
    words = re.findall(r"[a-z']+", question.lower())
    return intent == "general_support" and len(words) <= 10 and any(w in VAGUE_WORDS for w in words)


def unsupported_numbers(draft: str, context: str, question: str) -> List[str]:
    """Numbers in the answer that appear nowhere in the context or question (likely hallucinated).
    Step numbers like '4.' at the start of a list line are ignored (fixes an Expt 6 false alarm)."""
    nums = lambda t: set(re.findall(r"\d+(?:\.\d+)?", t))
    return sorted(nums(LIST_MARKER_RE.sub("", draft)) - nums(context) - nums(question))


# 4. The RAG service
class SupportRAG:
    """Created ONCE at API startup: embeddings, index and graph are reused for every request."""

    def __init__(self):
        if not os.path.exists(SOURCE_PDF):
            raise FileNotFoundError(f"Knowledge-base PDF not found: {SOURCE_PDF}")
        self.backend = make_backend()
        # fastembed = same MiniLM model on ONNX Runtime (no PyTorch): ~3x less RAM, fits 512 MB hosts.
        # Its output is already L2-normalized, so inner product == cosine similarity.
        # small batches keep peak RAM low if the index ever has to be built on a 512 MB host
        self.embeddings = FastEmbedEmbeddings(model_name=EMBED_MODEL, batch_size=16)
        self.vector_store, self.n_chunks = load_or_build_index(self.embeddings)
        self.graph = self._build_graph()

    def classify_intent(self, state: SupportState):
        q = state["question"].lower()
        intent = next((label for label, kws in INTENT_RULES if any(k in q for k in kws)), "general_support")
        update = {"intent": intent, "search_query": state["question"], "retry_count": 0}
        if is_vague(state["question"], intent):
            update["review_note"] = "ambiguous question"
        return update

    def retrieve(self, state: SupportState):
        results = self.vector_store.similarity_search_with_score(state["search_query"], k=TOP_K)
        docs = [d for d, _ in results]
        context = "\n\n".join(f"[{i + 1}] {d.page_content}" for i, d in enumerate(docs))
        sources = [f"{d.metadata.get('source')} (p.{d.metadata.get('page')}, {d.metadata.get('section')})"
                   for d in docs]
        update = {"context": context, "sources": sources}
        if state.get("top_score") is None:  
            _, top = self.vector_store.similarity_search_with_score(state["question"], k=1)[0]
            update["top_score"] = float(top)
            if top < MIN_RELEVANCE:
                update["review_note"] = f"question not covered by FAQ (similarity {top:.2f})"
        return update

    def draft_answer(self, state: SupportState):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"FAQ CONTEXT:\n{state.get('context', '')}\n\n"
                                        f"CUSTOMER QUESTION:\n{state['question']}"},
        ]
        draft = self.backend.generate(messages)         
        draft = re.split(r"\n\s*(?:CUSTOMER QUESTION|FAQ CONTEXT|Q:)", draft)[0].strip()
        draft = re.sub(r"^(ANSWER|A)\s*:\s*", "", draft)
        return {"draft": draft}

    def review_answer(self, state: SupportState):
        draft = state.get("draft", "").strip()
        if not draft or "INSUFFICIENT_CONTEXT" in draft.upper().replace(" ", "_"):
            return {"final_answer": "", "review_note": "model reported insufficient context"}
        bad = unsupported_numbers(draft, state.get("context", ""), state["question"])
        if bad:
            return {"final_answer": "", "review_note": f"numbers not in FAQ context: {bad}"}
        return {"final_answer": draft, "review_note": "grounded in retrieved FAQ", "used_fallback": False}

    def rewrite_query(self, state: SupportState):
        extra = CATEGORY_TERMS.get(state.get("intent", "general_support"), "customer support")
        return {"search_query": f"{state['question']} {extra}", "retry_count": state.get("retry_count", 0) + 1}

    def fallback(self, state: SupportState):
        text = CLARIFY_TEXT if state.get("review_note") == "ambiguous question" else FALLBACK_TEXT
        return {"final_answer": text, "used_fallback": True}

    @staticmethod
    def route_after_classify(state):
        return "clarify" if state.get("review_note") == "ambiguous question" else "retrieve"

    @staticmethod
    def route_after_retrieve(state):
        return "fallback" if state.get("top_score", 0.0) < MIN_RELEVANCE else "draft"

    @staticmethod
    def route_after_review(state):
        if state.get("final_answer"):
            return "finish"
        return "retry" if state.get("retry_count", 0) < 1 else "fallback"

    def _build_graph(self):
        g = StateGraph(SupportState)
        g.add_node("classify_intent", self.classify_intent)
        g.add_node("retrieve", self.retrieve)
        g.add_node("draft_answer", self.draft_answer)
        g.add_node("review_answer", self.review_answer)
        g.add_node("rewrite_query", self.rewrite_query)
        g.add_node("fallback", self.fallback)

        g.add_edge(START, "classify_intent")
        g.add_conditional_edges("classify_intent", self.route_after_classify,
                                {"clarify": "fallback", "retrieve": "retrieve"})
        g.add_conditional_edges("retrieve", self.route_after_retrieve,
                                {"fallback": "fallback", "draft": "draft_answer"})
        g.add_edge("draft_answer", "review_answer")
        g.add_conditional_edges("review_answer", self.route_after_review,
                                {"finish": END, "retry": "rewrite_query", "fallback": "fallback"})
        g.add_edge("rewrite_query", "retrieve")
        g.add_edge("fallback", END)
        return g.compile()

    def answer(self, question: str) -> dict:
        r = self.graph.invoke({"question": question})
        return {
            "answer": r.get("final_answer", ""),
            "intent": r.get("intent", "general_support"),
            "used_fallback": bool(r.get("used_fallback", False)),
            "review_note": r.get("review_note", ""),
            "retries": r.get("retry_count", 0),
            "top_similarity": round(r.get("top_score", 0.0) or 0.0, 3),
            "sources": r.get("sources", []),
        }
