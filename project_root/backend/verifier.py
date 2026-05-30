import json
import os
from urllib.request import Request, urlopen
from urllib.error import URLError

from . import config
from llm.json_schema import LLMOutput

_use_classifier: bool | None = None


def _classifier_is_available() -> bool:
    """Check whether the trained sklearn classifier model exists on disk."""
    try:
        from .classifier import is_model_available
        return is_model_available()
    except Exception:
        return False


def enable_classifier(flag: bool = True) -> None:
    """Enable / disable the trained classifier as a heuristic_verify replacement."""
    global _use_classifier
    _use_classifier = flag


def _should_use_classifier() -> bool:
    global _use_classifier
    if _use_classifier is None:
        _use_classifier = _classifier_is_available()
    return _use_classifier


DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
DEEPSEEK_MODEL = "deepseek-chat"
DEEPSEEK_TIMEOUT = 30
DIRECT_PROMPT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "llm", "prompts", "direct_verification_prompt.txt"
)
PROMPT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "llm", "prompts", "verification_prompt.txt"
)
BINARY_GATE_PROMPT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "llm", "prompts", "binary_gate_prompt.txt"
)


TITLE_W = 0.18
AUTHOR_W = 0.25
YEAR_W = 0.11
VENUE_W = 0.05
DOI_W = 0.10
SEMANTIC_W = 0.31


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


def _detect_metadata_issues(candidate: dict, parsed_citation: dict | None = None) -> list[str]:
    issues = []
    year_sim = candidate.get("year_similarity", 0.0)
    venue_sim = candidate.get("venue_similarity", 0.0)
    doi_sim = candidate.get("doi_similarity", 0.0)

    if _metadata_present(candidate, "year"):
        if year_sim < 0.1:
            p_year = (parsed_citation or {}).get("year")
            db_year = candidate.get("year")
            if db_year is not None and p_year is not None and abs(int(db_year) - int(p_year)) > 3:
                issues.append("year differs significantly")
        elif year_sim < 0.5:
            issues.append("year differs")

    if _metadata_present(candidate, "venue"):
        db_venue = str(candidate.get("venue", "")).strip()
        has_problematic_venue = False
        venue_len = len(db_venue)
        if venue_len > 30 and " " not in db_venue[:30]:
            has_problematic_venue = True
        elif venue_sim < 0.05:
            has_problematic_venue = True
        if has_problematic_venue:
            issues.append("venue differs")

    if _metadata_present(candidate, "doi") and doi_sim < 0.5:
        p_doi = ((parsed_citation or {}).get("doi") or "").strip().lower()
        db_doi = str(candidate.get("doi", "")).strip().lower()
        if p_doi and db_doi and p_doi != db_doi:
            issues.append("DOI mismatch")

    return issues


def heuristic_verify(top_candidate: dict) -> dict:
    if not top_candidate:
        return {
            "label": "HALLUCINATED",
            "confidence": 0.0,
            "reason": "No candidate provided",
        }

    score = _component_score(top_candidate)

    if score >= 0.70:
        return {
            "label": "VALID",
            "confidence": score,
            "reason": "Weighted component score exceeds VALID threshold",
        }

    if score >= 0.30:
        return {
            "label": "PARTIALLY_VALID",
            "confidence": score,
            "reason": "Weighted component score in PARTIALLY_VALID range",
        }

    return {
        "label": "HALLUCINATED",
        "confidence": score,
        "reason": "Weighted component score below minimum threshold",
    }


def verify_top_candidate(
    top_candidate: dict,
    parsed_citation: dict | None = None,
    exact_title_match: int = 0,
    exact_doi_match: int = 0,
) -> dict:
    """
    Verify a top candidate using the best available method.

    Layered approach:
      1. If paper is in DB (exact_title_match), it's at minimum PARTIALLY_VALID.
         - Detect specific corruption types to decide VALID vs PARTIALLY_VALID.
      2. If not in DB, use classifier or heuristic for HALL vs PARTIALLY_VALID.
         - Never classify as VALID without DB evidence.
    """
    if not top_candidate:
        return heuristic_verify(top_candidate)

    title_sim = top_candidate.get("fuzzy_score", 0.0) / 100.0
    author_sim = top_candidate.get("author_similarity", 0.0)
    year_sim = top_candidate.get("year_similarity", 0.0)
    venue_sim = top_candidate.get("venue_similarity", 0.0)
    doi_sim = top_candidate.get("doi_similarity", 0.0)
    metadata_score = top_candidate.get("metadata_score", 0.0)
    final_score = top_candidate.get("final_score", 0.0)
    sem_raw = top_candidate.get("semantic_score", -1.0)
    sem_sim = 1.0 / (1.0 + sem_raw) if sem_raw >= 0 else 0.0
    score = _component_score(top_candidate)

    # Layer 1: Paper is in DB → can't be HALLUCINATED
    if exact_title_match:
        corruptions = []

        # Year way off → strong signal
        if year_sim == 0.0:
            corruptions.append("year_wrong")
        # Year shifted by 1 is ambiguous (could be preprint vs pub year) → weak signal
        elif year_sim == 0.5:
            corruptions.append("year_shifted")

        # Title corruption
        if title_sim < 0.85:
            corruptions.append("title_typo")

        # Venue corruption (stricter threshold — venue format often differs)
        if venue_sim < 0.05 and metadata_score < 0.15:
            corruptions.append("venue_mismatch")

        # Author corruption (only if authors present and clearly mismatching)
        if author_sim > 0 and author_sim < 0.20 and metadata_score < 0.25:
            corruptions.append("author_mismatch")

        # year_shifted alone is weak — only count if another corruption exists
        # (DB often stores preprint year, citation has pub year → ±1 is normal)
        if corruptions == ["year_shifted"]:
            return {
                "label": "VALID",
                "confidence": 0.85,
                "reason": "Paper in DB — minor year difference (±1) likely preprint vs publication",
            }

        if len(corruptions) >= 2:
            return {
                "label": "PARTIALLY_VALID",
                "confidence": round(0.80 - 0.05 * len(corruptions), 4),
                "reason": f"Paper in DB but multiple corruptions: {', '.join(corruptions)}",
            }
        elif len(corruptions) == 1:
            return {
                "label": "PARTIALLY_VALID",
                "confidence": 0.70,
                "reason": f"Paper in DB but {corruptions[0]}",
            }

        # Use classifier if available for final decision
        if _should_use_classifier() and parsed_citation is not None:
            try:
                from .classifier import classify
                result = classify(
                    top_candidate, parsed_citation,
                    exact_title_match=exact_title_match,
                    exact_doi_match=exact_doi_match,
                )
                label = result.get("label", "VALID")
                if label == "HALLUCINATED":
                    return {
                        "label": "PARTIALLY_VALID",
                        "confidence": 0.65,
                        "probabilities": result.get("probabilities", {}),
                        "reason": "Paper in DB (exact title match) — paper exists but classifier uncertain",
                    }
                if label == "PARTIALLY_VALID":
                    return {
                        "label": "VALID",
                        "confidence": 0.85,
                        "probabilities": result.get("probabilities", {}),
                        "reason": "Paper in DB with no corruption signals detected",
                    }
                return result
            except Exception:
                pass

        return {
            "label": "VALID",
            "confidence": round(max(0.85, score), 4),
            "reason": "Paper in DB (exact title match) with no corruption signals",
        }

    # Layer 2: Not in DB → fall back to heuristic/classifier
    if _should_use_classifier() and parsed_citation is not None:
        try:
            from .classifier import classify
            result = classify(
                top_candidate, parsed_citation,
                exact_title_match=exact_title_match,
                exact_doi_match=exact_doi_match,
            )
            label = result.get("label", "HALLUCINATED")
            if label == "VALID":
                result["label"] = "PARTIALLY_VALID"
                result["confidence"] = round(result.get("confidence", 0.0) * 0.7, 4)
                result["reason"] = "No exact DB match — downgraded from VALID to PARTIALLY_VALID"
            return result
        except Exception:
            pass

    # Heuristic for non-DB papers
    # Need strong signal for PARTIALLY_VALID; default is HALLUCINATED
    if title_sim >= 0.85 and final_score >= 50:
        if author_sim >= 0.3 and year_sim >= 0.5:
            return {
                "label": "PARTIALLY_VALID",
                "confidence": round(min(0.55, score), 4),
                "reason": "Strong retrieval signal without DB match",
            }
    if title_sim >= 0.70 and author_sim >= 0.3 and metadata_score >= 0.25:
        return {
            "label": "PARTIALLY_VALID",
            "confidence": round(min(0.45, score), 4),
            "reason": "Moderate retrieval signal without DB match",
        }

    return {
        "label": "HALLUCINATED",
        "confidence": round(max(0.0, score * 0.4), 4),
        "reason": "No DB match — likely hallucinated",
    }

    title_sim = top_candidate.get("fuzzy_score", 0.0) / 100.0
    author_sim = top_candidate.get("author_similarity", 0.0)
    year_sim = top_candidate.get("year_similarity", 0.0)
    final_score = top_candidate.get("final_score", 0.0)
    score = _component_score(top_candidate)

    # Rule 1: Strong title + author match -> VALID (only if metadata issues are minor)
    if title_sim >= 0.95 and author_sim >= 0.70:
        issues = _detect_metadata_issues(top_candidate, parsed_citation)
        critical_issues = [i for i in issues if "significantly" in i or "DOI mismatch" in i]
        if len(critical_issues) == 0 and len(issues) <= 1:
            return {
                "label": "VALID",
                "confidence": round(max(0.85, score), 4),
                "reason": "Title and author match database record",
            }
        if issues:
            return {
                "label": "PARTIALLY_VALID",
                "confidence": round(score * 0.85, 4),
                "reason": f"Title and authors match but {'; '.join(issues)}",
            }
        return {
            "label": "VALID",
            "confidence": round(max(0.85, score), 4),
            "reason": "Title, author, and metadata all match database record",
        }

    # Rule 2: Title-only match -> PARTIALLY_VALID
    if title_sim >= 0.95:
        return {
            "label": "PARTIALLY_VALID",
            "confidence": round(score * 0.8, 4),
            "reason": "Title matches exactly but authors, year, or venue do not match database record",
        }

    # Rule 3: High title similarity (>= 0.85) needs BOTH author AND year signals (AND gate)
    if title_sim >= 0.85:
        if author_sim >= 0.3 and year_sim >= 0.5:
            return {
                "label": "PARTIALLY_VALID",
                "confidence": round(score * 0.7, 4),
                "reason": "Partial match with author and year signals",
            }

    # Rule 4: Medium title similarity (0.70-0.85) with strong combined signal
    if title_sim >= 0.70 and final_score >= 50:
        if author_sim >= 0.3 and year_sim >= 0.5:
            return {
                "label": "PARTIALLY_VALID",
                "confidence": round(score * 0.55, 4),
                "reason": "Medium title match with corroborating author and year signals",
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
            "reason": "LLM verification unavailable -- no database or API match found",
        }

    try:
        output = LLMOutput(**raw)
        return output.model_dump()
    except Exception:
        return {
            "label": "HALLUCINATED",
            "confidence": 0.0,
            "reason": "LLM verification failed -- no database or API match found",
        }


def _load_binary_gate_template() -> str:
    with open(BINARY_GATE_PROMPT_PATH, "r") as f:
        return f.read()


def llm_binary_gate(parsed_citation: dict) -> bool:
    """Simple LLM binary gate: does this paper EXIST?
    Returns True if LLM thinks paper is REAL, False if FAKE.
    Falls back to True on API failure (let deterministic check decide).
    """
    template = _load_binary_gate_template()
    prompt = template.format(
        parsed_title=parsed_citation.get("title", ""),
        parsed_authors=parsed_citation.get("authors", ""),
        parsed_year=parsed_citation.get("year", "unknown"),
        parsed_venue=parsed_citation.get("venue", ""),
        parsed_doi=parsed_citation.get("doi", ""),
    )

    api_key = config.DEEPSEEK_API_KEY
    if not api_key:
        return True  # no API → assume real, let deterministic check decide

    payload = json.dumps({
        "model": DEEPSEEK_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
    }).encode("utf-8")

    try:
        req = Request(
            DEEPSEEK_API_URL,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
        )
        with urlopen(req, timeout=DEEPSEEK_TIMEOUT) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            content = data["choices"][0]["message"]["content"].strip().upper()
            return "REAL" in content
    except (URLError, json.JSONDecodeError, KeyError, OSError):
        return True  # API fail → assume real, let deterministic check decide
