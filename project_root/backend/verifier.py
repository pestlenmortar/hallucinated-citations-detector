import json
import os
from urllib.request import Request, urlopen
from urllib.error import URLError

from llm.json_schema import LLMOutput

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5:3b"
OLLAMA_TIMEOUT = 30
PROMPT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "llm", "prompts", "verification_prompt.txt"
)


def heuristic_verify(top_candidate: dict) -> dict:
    if not top_candidate:
        return {
            "label": "HALLUCINATED",
            "confidence": 0.0,
            "reason": "No candidate provided",
        }

    if top_candidate.get("fuzzy_score", 0.0) == 100.0:
        return {
            "label": "VALID",
            "confidence": 1.0,
            "reason": "Exact title match found in database",
        }

    fuzzy = top_candidate.get("fuzzy_score", 0.0)
    semantic = top_candidate.get("semantic_score", 0.0)
    final = top_candidate.get("final_score", 0.0)
    author_sim = top_candidate.get("author_similarity", 0.0)
    year_sim = top_candidate.get("year_similarity", 0.0)

    title_sim = fuzzy / 100.0
    year_diff_ok = year_sim >= 0.5

    if title_sim > 0.90 and author_sim > 0.70 and year_diff_ok:
        conf = round(0.4 * title_sim + 0.3 * author_sim + 0.3 * year_sim, 4)
        return {
            "label": "VALID",
            "confidence": conf,
            "reason": "Title, author, and year all match within thresholds",
        }

    if (title_sim > 0.70 or final > 60) and (author_sim > 0.3 or year_diff_ok):
        conf = round(0.5 * title_sim + 0.3 * author_sim + 0.2 * year_sim, 4)
        return {
            "label": "PARTIALLY_VALID",
            "confidence": min(conf, 0.85),
            "reason": "Partial match found but some metadata is off",
        }

    return {
        "label": "HALLUCINATED",
        "confidence": round(max(0.0, title_sim * 0.3), 4),
        "reason": "Candidate does not meet minimum similarity thresholds",
    }


def _format_candidates(top_candidates: list) -> str:
    lines = []
    for i, c in enumerate(top_candidates[:5], 1):
        lines.append(
            f"  {i}. paper_id={c.get('paper_id')} "
            f"title=\"{c.get('title', '')}\" "
            f"fuzzy_score={c.get('fuzzy_score'):.2f} "
            f"semantic_score={c.get('semantic_score'):.4f} "
            f"author_sim={c.get('author_similarity'):.2f} "
            f"year_sim={c.get('year_similarity'):.2f} "
            f"venue_sim={c.get('venue_similarity'):.2f} "
            f"final_score={c.get('final_score'):.2f}"
        )
    return "\n".join(lines) if lines else "  (none)"


def _load_prompt_template() -> str:
    with open(PROMPT_PATH, "r") as f:
        return f.read()


def _call_ollama(prompt: str) -> dict | None:
    payload = json.dumps(
        {"model": OLLAMA_MODEL, "prompt": prompt, "format": "json", "stream": False}
    ).encode("utf-8")
    req = Request(
        OLLAMA_URL, data=payload, headers={"Content-Type": "application/json"}
    )
    try:
        with urlopen(req, timeout=OLLAMA_TIMEOUT) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return json.loads(data["response"])
    except (URLError, json.JSONDecodeError, KeyError, OSError):
        return None


def llm_verify(top_candidates: list, parsed_citation: dict) -> dict:
    template = _load_prompt_template()
    candidate_table = _format_candidates(top_candidates)
    prompt = template.format(
        parsed_title=parsed_citation.get("title", ""),
        parsed_authors=parsed_citation.get("authors", ""),
        parsed_year=parsed_citation.get("year", "unknown"),
        parsed_venue=parsed_citation.get("venue", ""),
        parsed_doi=parsed_citation.get("doi", ""),
        candidate_table=candidate_table,
    )

    raw = _call_ollama(prompt)
    if raw is None:
        first = top_candidates[0] if top_candidates else {}
        return heuristic_verify(first)

    try:
        output = LLMOutput(**raw)
        return output.model_dump()
    except Exception:
        first = top_candidates[0] if top_candidates else {}
        return heuristic_verify(first)
