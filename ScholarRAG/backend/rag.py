from dotenv import load_dotenv
load_dotenv()

import os
import io
import certifi
import requests
import anthropic
import pdfplumber
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from sentence_transformers import CrossEncoder

_qdrant_base   = os.path.join(os.path.dirname(__file__), "..", "qdrant_local")
_qdrant_nested = os.path.join(_qdrant_base, "qdrant_local")
QDRANT_PATH    = _qdrant_nested if os.path.isdir(_qdrant_nested) else _qdrant_base
COLLECTION  = "papers"
MODEL_NAME  = "all-MiniLM-L6-v2"

SYSTEM_PROMPT = """You are ScholarRAG, a research assistant with strong knowledge of machine learning, computer science, and academic research.

Your job is to actually answer the user's question — not to summarize papers at them.

Lead with your own understanding. The papers and code snippets provided are supplementary — cite them when they add something specific, ignore them if they're off-topic. Never force-fit an abstract into your answer just because it was retrieved.

How to respond:
- Answer the actual question first, from your own knowledge, in plain language
- Use the retrieved papers to add depth, a citation, or a concrete example — not as the backbone of your entire reply
- If code was fetched (README, source file): pull out the relevant snippet, show it, explain what it actually does
- If a PDF was uploaded: treat it as the primary document and answer relative to its content specifically
- Short question = short answer. Long technical question = go deep. Match the energy of what was asked
- No forced structure. Write like you're explaining to a smart colleague, not filling out a form
- No filler. No "Great question!", no "Certainly!", no restating the question

If the user asks which AI model or technology powers YOU specifically — respond with: "Sorry, that's outside the scope of this application. I'm here to help you explore academic research." Do NOT apply this to any research question about AI, ML, or code."""

_embeddings    = None
_qdrant        = None
_client        = None
_async_client  = None
_cross_encoder = None

_memories: dict[str, list] = {}


def init():
    global _embeddings, _qdrant, _client, _async_client, _cross_encoder

    _embeddings    = HuggingFaceEmbeddings(model_name=MODEL_NAME)
    _qdrant        = QdrantClient(path=QDRANT_PATH)
    _client        = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    _async_client  = anthropic.AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    _cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def _extract_pdf_text(pdf_bytes: bytes) -> str:
    text_parts = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                text_parts.append(t)
    return "\n\n".join(text_parts)


def _search(query_text: str, k: int = 5):
    vector = _embeddings.embed_query(query_text)
    return _qdrant.query_points(
        collection_name=COLLECTION,
        query=vector,
        limit=k,
    ).points


def _get_memory(session_id: str) -> list:
    if session_id not in _memories:
        _memories[session_id] = []
    return _memories[session_id]


def clear_memory(session_id: str):
    if session_id in _memories:
        del _memories[session_id]


def _github_links(payload: dict) -> list[str]:
    repos = payload.get("github_repos", [])
    if repos and isinstance(repos, list):
        return [r["url"] if isinstance(r, dict) else r for r in repos if r]
    single = payload.get("repo_url", "")
    return [single] if single else []



FETCH_URL_TOOL = {
    "name": "fetch_url",
    "description": (
        "Fetch the content of a URL. Use this to read GitHub repositories, "
        "READMEs, raw code files, or arXiv papers when you need to understand "
        "an implementation in depth. For GitHub repo URLs, prefer fetching the "
        "README or specific Python files."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "The URL to fetch."}
        },
        "required": ["url"],
    },
}


def _execute_fetch(url: str) -> str:
    try:
        if "github.com" in url and "/blob/" in url:
            url = url.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")
        headers = {"User-Agent": "ScholarRAG/1.0"}
        gh_token = os.getenv("GITHUB_TOKEN", "")
        if ("github.com" in url or "raw.githubusercontent.com" in url) and gh_token:
            headers["Authorization"] = f"token {gh_token}"
        r = requests.get(url, headers=headers, timeout=15, verify=certifi.where())
        if r.status_code != 200:
            return f"HTTP {r.status_code} — could not fetch URL"
        return r.text[:6000]
    except Exception as e:
        return f"Error fetching URL: {e}"


def _classify_query(question: str) -> str:
    resp = _client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=10,
        messages=[{
            "role": "user",
            "content": (
                f"Classify this query as exactly one word — either 'research' or 'conversational'.\n\n"
                f"- 'research': asks about papers, topics, concepts, code, authors, findings\n"
                f"- 'conversational': greetings, meta-questions about the assistant, thanks, small talk\n\n"
                f"Query: {question}\n\nAnswer:"
            ),
        }],
    )
    label = resp.content[0].text.strip().lower()
    return "research" if "research" in label else "conversational"


def _hyde_query(question: str) -> str:
    try:
        resp = _client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=80,
            messages=[{
                "role": "user",
                "content": (
                    f"Write a 2-3 sentence academic paper abstract that would directly answer this question:\n\n"
                    f"{question}\n\n"
                    f"Write only the abstract text. No title, no explanation, no preamble."
                ),
            }],
        )
        return resp.content[0].text.strip()
    except Exception:
        return question


def _prepare_query(question: str, session_id: str, pdf_bytes: bytes = None, preloaded_history: list = None):
    if session_id not in _memories and preloaded_history:
        _memories[session_id] = preloaded_history
    history = _get_memory(session_id)
    sources = []

    if True:
        pdf_text    = _extract_pdf_text(pdf_bytes) if pdf_bytes else None
        search_text = f"{question} {pdf_text[:1500]}" if pdf_text else question

        SCORE_THRESHOLD = 0.35
        TARGET          = 3

        candidates   = _search(search_text, k=8)
        filtered     = [r for r in candidates if r.score >= SCORE_THRESHOLD]
        with_code    = [r for r in filtered if _github_links(r.payload)]
        without_code = [r for r in filtered if not _github_links(r.payload)]

        if with_code:
            results = (with_code[:1] + without_code[:TARGET - 1])[:TARGET]
        else:
            results = without_code[:TARGET]

        context_parts = []
        for r in results[:3]:
            p     = r.payload
            lines = [f"Title: {p.get('title', '')}"]
            year  = (p.get("published") or "")[:4]
            if year:
                lines.append(f"Year: {year}")
            lines.append(f"Abstract: {p.get('summary', '')}")
            github = _github_links(p)
            if github:
                lines.append(f"Implementation (GitHub): {', '.join(github)}")
            context_parts.append("\n".join(lines))

        top_with_code = next((r for r in results if _github_links(r.payload)), None)
        code_note = ""
        if top_with_code:
            p     = top_with_code.payload
            repos = _github_links(p)
            code_note = (
                f"\n\nBest paper with code: \"{p.get('title', '')}\" — "
                f"GitHub: {', '.join(repos)}"
                f"\nAlways mention this implementation link explicitly in your answer."
            )

        qdrant_context = "\n\n---\n\n".join(context_parts) + code_note

        if pdf_text:
            user_msg = (
                f"Question: {question}\n\n"
                f"The user uploaded a PDF — answer relative to this document specifically:\n\n{pdf_text}\n\n"
                f"---\nRelated papers for additional context (use only if relevant):\n{qdrant_context}"
            )
        else:
            user_msg = (
                f"Question: {question}\n\n"
                f"---\nRelated papers for reference (cite when useful, don't force-fit):\n{qdrant_context}"
            )

        for r in results:
            p = r.payload
            sources.append({
                "title":           p.get("title", ""),
                "arxiv_id":        p.get("arxiv_id", ""),
                "authors":         p.get("authors", []),
                "published":       p.get("published", ""),
                "pdf_url":         p.get("pdf_url", ""),
                "category":        p.get("category", ""),
                "summary":         (p.get("summary", "")[:200] + "...") if p.get("summary") else "",
                "github_repos":    _github_links(p),
                "has_code":        len(_github_links(p)) > 0,
                "citations":       p.get("citations", 0),
                "score":           round(r.score, 3),
                "journal":         p.get("journal", ""),
                "doi":             p.get("doi", ""),
                "fields_of_study": p.get("fields_of_study", []),
            })
    api_messages = list(history)
    api_messages.append({"role": "user", "content": user_msg})
    return api_messages, sources, history


def query_rag(question: str, session_id: str, pdf_bytes: bytes = None, preloaded_history: list = None, model: str = "claude-sonnet-4-6") -> dict:
    api_messages, sources, history = _prepare_query(question, session_id, pdf_bytes, preloaded_history)

    max_tool_rounds = 3
    answer = ""

    for _ in range(max_tool_rounds + 1):
        response = _client.messages.create(
            model=model,
            max_tokens=2048,
            cache_control={"type": "ephemeral"},
            system=[{"type": "text", "text": SYSTEM_PROMPT}],
            messages=api_messages,
            tools=[FETCH_URL_TOOL],
        )

        if response.stop_reason == "tool_use":
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    content = _execute_fetch(block.input.get("url", ""))
                    tool_results.append({
                        "type":        "tool_result",
                        "tool_use_id": block.id,
                        "content":     content,
                    })
            api_messages.append({"role": "assistant", "content": response.content})
            api_messages.append({"role": "user",      "content": tool_results})
        else:
            answer = next((b.text for b in response.content if hasattr(b, "text")), "")
            break

    history.append({"role": "user",      "content": question})
    history.append({"role": "assistant", "content": answer})

    return {"answer": answer, "sources": sources}


async def query_rag_stream(question: str, session_id: str, pdf_bytes: bytes = None, preloaded_history: list = None, model: str = "claude-sonnet-4-6"):
    import json
    import asyncio
    yield f"data: {json.dumps({'type': 'status', 'text': 'Searching papers...'})}\n\n"
    api_messages, sources, history = await asyncio.to_thread(_prepare_query, question, session_id, pdf_bytes, preloaded_history)
    yield f"data: {json.dumps({'type': 'status', 'text': 'Generating answer...'})}\n\n"

    max_tool_rounds = 3
    answer = ""

    for _ in range(max_tool_rounds + 1):
        async with _async_client.messages.stream(
            model=model,
            max_tokens=2048,
            system=[{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
            messages=api_messages,
            tools=[FETCH_URL_TOOL],
        ) as stream:
            async for text in stream.text_stream:
                answer += text
                yield f"data: {json.dumps({'type': 'token', 'text': text})}\n\n"
            final_msg = await stream.get_final_message()

        if final_msg.stop_reason == "tool_use":
            tool_results = []
            for block in final_msg.content:
                if block.type == "tool_use":
                    content = await asyncio.to_thread(_execute_fetch, block.input.get("url", ""))
                    tool_results.append({
                        "type":        "tool_result",
                        "tool_use_id": block.id,
                        "content":     content,
                    })
            api_messages.append({"role": "assistant", "content": final_msg.content})
            api_messages.append({"role": "user",      "content": tool_results})
        else:
            break

    history.append({"role": "user",      "content": question})
    history.append({"role": "assistant", "content": answer})

    yield f"data: {json.dumps({'type': 'done', 'sources': sources})}\n\n"
