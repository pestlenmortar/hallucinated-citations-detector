import json
import os
from urllib.request import Request, urlopen
from urllib.error import URLError

from . import config
from llm.json_schema import LLMOutput

DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
DEEPSEEK_MODEL = "deepseek-chat"
DEEPSEEK_TIMEOUT = 30
DIRECT_PROMPT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "llm", "prompts", "direct_verification_prompt.txt"
)
PROMPT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "llm", "prompts", "verification_prompt.txt"
)


TITLE_W = 0.18
AUTHOR_W = 0.21
YEAR_W = 0.11
VENUE_W = 0.05
DOI_W = 0.10
SEMANTIC_W = 0.35


def _component_score(candidate: dict) -> float:
    title_sim = candidate.get("fuzzy_score", 0.0) / 100.0
    author_sim = candidate.get("author_similarity", 0.0)
    year_sim = candidate.get("year_similarity", 0.0)
    venue_sim = candidate.get("venue_similarity", 0.0)
    doi_sim = candidate.get("doi_similarity", 0.0)
    sem_raw = candidate.get("semantic_score", -1.0)
    sem_sim = 1.0 / (1.0 + sem_raw) if sem_raw >= 0 else 0.0

    return round(
        TITLE_W * title_sim
        + AUTHOR_W * author_sim
        + YEAR_W * year_sim
        + VENUE_W * venue_sim
        + DOI_W * doi_sim
        + SEMANTIC_W * sem_sim,
        4,
    )


def _metadata_present(candidate: dict, field: str) -> bool:
    val = candidate.get(field, "")
    return bool(val and str(val).strip())


def _detect_metadata_issues(candidate: dict) -> list[str]:
    issues = []
    if _metadata_present(candidate, "year") and candidate.get("year_similarity", 0.0) < 0.3:
        issues.append("year differs")
    if _metadata_present(candidate, "venue") and candidate.get("venue_similarity", 0.0) == 0.0:
        issues.append("venue differs")
    if _metadata_present(candidate, "doi") and candidate.get("doi_similarity", 0.0) < 0.8:
        issues.append("DOI mismatch")
    return issues


def heuristic_verify(top_candidate: dict) -> dict:
    if not top_candidate:
        return {
            "label": "HALLUCINATED",
            "confidence": 0.0,
            "reason": "No candidate provided",
        }

    title_sim = top_candidate.get("fuzzy_score", 0.0) / 100.0
    author_sim = top_candidate.get("author_similarity", 0.0)
    year_sim = top_candidate.get("year_similarity", 0.0)
    final_score = top_candidate.get("final_score", 0.0)
    score = _component_score(top_candidate)

    if title_sim >= 0.95 and author_sim >= 0.70:
        issues = _detect_metadata_issues(top_candidate)
        if not issues:
            return {
                "label": "VALID",
                "confidence": round(max(0.90, score), 4),
                "reason": "Title, author, and metadata all match database record",
            }
        return {
            "label": "PARTIALLY_VALID",
            "confidence": round(score * 0.85, 4),
            "reason": f"Title and authors match but {'; '.join(issues)}",
        }

    if title_sim >= 0.95:
        return {
            "label": "PARTIALLY_VALID",
            "confidence": round(score * 0.8, 4),
            "reason": "Title matches exactly but authors, year, or venue do not match database record",
        }

    if title_sim >= 0.85 or final_score >= 50:
        if author_sim >= 0.3 and year_sim >= 0.5:
            return {
                "label": "PARTIALLY_VALID",
                "confidence": round(score, 4),
                "reason": "Partial match found but some metadata is off",
            }

    return {
        "label": "HALLUCINATED",
        "confidence": round(max(0.0, score * 0.3), 4),
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
            f"doi_sim={c.get('doi_similarity'):.2f} "
            f"final_score={c.get('final_score'):.2f}"
        )
    return "\n".join(lines) if lines else "  (none)"


def _load_prompt_template() -> str:
    with open(PROMPT_PATH, "r") as f:
        return f.read()


def _call_deepseek(prompt: str) -> dict | None:
    api_key = config.DEEPSEEK_API_KEY
    if not api_key:
        return None
    payload = json.dumps({
        "model": DEEPSEEK_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "response_format": {"type": "json_object"},
        "temperature": 0.0,
    }).encode("utf-8")
    req = Request(
        DEEPSEEK_API_URL,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    try:
        with urlopen(req, timeout=DEEPSEEK_TIMEOUT) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            content = data["choices"][0]["message"]["content"]
            return json.loads(content)
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

    raw = _call_deepseek(prompt)
    if raw is None:
        first = top_candidates[0] if top_candidates else {}
        return heuristic_verify(first)

    try:
        output = LLMOutput(**raw)
        return output.model_dump()
    except Exception:
        first = top_candidates[0] if top_candidates else {}
        return heuristic_verify(first)


def _load_direct_prompt_template() -> str:
    with open(DIRECT_PROMPT_PATH, "r") as f:
        return f.read()


def llm_verify_direct(parsed_citation: dict) -> dict:
    template = _load_direct_prompt_template()
    prompt = template.format(
        parsed_title=parsed_citation.get("title", ""),
        parsed_authors=parsed_citation.get("authors", ""),
        parsed_year=parsed_citation.get("year", "unknown"),
        parsed_venue=parsed_citation.get("venue", ""),
        parsed_doi=parsed_citation.get("doi", ""),
    )

    raw = _call_deepseek(prompt)
    if raw is None:
        return {
            "label": "HALLUCINATED",
            "confidence": 0.0,
            "reason": "LLM verification unavailable — no database or API match found",
        }

    try:
        output = LLMOutput(**raw)
        return output.model_dump()
    except Exception:
        return {
            "label": "HALLUCINATED",
            "confidence": 0.0,
            "reason": "LLM verification failed — no database or API match found",
        }
