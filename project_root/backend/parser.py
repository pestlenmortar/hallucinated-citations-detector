import re

from .models import ParsedCitation
from .normalization import normalize_title

YEAR_RE = r"\b((?:1[89]\d{2}|20[0-2]\d))\b"
DOI_RE = (
    r"(?:https?://(?:dx\.)?doi\.org/|doi:?\s*)?"
    r"(10\.\d{4,}(?:\.\d+)*/[^\s,;)\]}>]+)"
)

# Matches APA/Harvard author lists at the start of a string
AUTHOR_APA_RE = (
    r"(?:[A-Z][a-z]+,\s[A-Z](?:\.[A-Z])?\.?"
    r"(?:\s*[&,]\s*(?:[A-Z][a-z]+,\s[A-Z](?:\.[A-Z])?\.?"
    r"|et\s*al\.?))?\s*[,;.]?\s*)+"
)

# Matches IEEE-style "J. Smith" at the start
AUTHOR_IEE_RE = (
    r"(?:[A-Z]\.\s[A-Z][a-z]+(?:\s+[A-Z]\.(?:\s+[A-Z][a-z]+)?)?"
    r"(?:\s+(?:and|&)\s+[A-Z]\.\s[A-Z][a-z]+(?:\s+[A-Z]\.)?)?)\s*"
)

# Matches venue-like segments: journal names starting with a capital
VENUE_RE = (
    r"(?:(?:Journal|Proceedings|Conference|Transactions|International|"
    r"Annals|Review|Letters|Communications|Advances|Research|"
    r"British|European|IEEE|ACM|Springer|Elsevier|Nature|Science|PLOS|"
    r"Frontiers)\s[A-Z][a-zA-Z]+)"
)


def _find_doi(text: str) -> str:
    m = re.search(DOI_RE, text, re.IGNORECASE)
    return m.group(1).rstrip(".,;") if m else ""


def _find_year(text: str) -> int | None:
    m = re.search(YEAR_RE, text)
    return int(m.group(1)) if m else None


def _find_authors(text: str) -> str:
    m = re.match(AUTHOR_APA_RE, text)
    if m:
        return m.group(0).strip().rstrip(".,; ")
    m = re.match(AUTHOR_IEE_RE, text)
    if m:
        return m.group(0).strip().rstrip(".,; ")
    return ""


def _extract_title_after_year(after_year: str) -> str:
    """Try to pull the title from text following the year (APA style)."""
    # Split on the first period followed by a space and uppercase
    m = re.match(r"\.?\s*([A-Z][^.]*?\.)(?=\s+[A-Z])", after_year)
    if m:
        candidate = m.group(1).strip().rstrip(".")
        if len(candidate) > 10 and candidate.count(" ") < 40:
            return candidate
    # Fallback: everything up to the first period
    first_period = after_year.find(".")
    if first_period != -1:
        return after_year[:first_period].strip().lstrip(".")
    return after_year.strip()


def _extract_title_before_year(before_year: str, authors: str) -> str:
    """Extract title from text before the year (IEEE style, or
    title-first formats)."""
    remaining = before_year
    if authors:
        remaining = before_year[len(authors):].strip().lstrip(".,; ")
    # Check for quoted title (IEEE: "Title")
    qm = re.search(r'"([^"]+)"', remaining)
    if qm:
        return qm.group(1)
    # First sentence as title
    period_idx = remaining.find(".")
    if period_idx != -1:
        return remaining[:period_idx].strip()
    return remaining.strip()


def _find_venue(text: str, doi: str) -> str:
    if doi:
        idx = text.lower().find(doi.lower())
        if idx != -1:
            text = text[:idx]
    text = re.sub(r"\b\d+\s*\(\d+\)\s*:?\s*\d+[-–]\d+\b", "", text)
    text = re.sub(r"(?:vol|no|pp)\.?\s*\d+[^\s,;]*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"[;:,.\s]+$", "", text.strip())
    return text


def parse_citation(raw: str) -> ParsedCitation:
    text = raw.strip()
    doi = _find_doi(text)
    year = _find_year(text)
    text_no_doi = re.sub(DOI_RE, "", text, flags=re.IGNORECASE).strip()

    before_year = text_no_doi
    after_year = ""
    if year is not None:
        ypos = text_no_doi.find(str(year))
        if ypos != -1:
            before_year = text_no_doi[:ypos].strip().rstrip(".,; (")
            after_year = text_no_doi[ypos + 4:].strip().lstrip(".,;: )")

    authors = _find_authors(before_year)

    title = ""
    try_before = False

    if authors and after_year and len(after_year) > 10:
        # APA-like: authors found at start, title likely after year
        raw_title = _extract_title_after_year(after_year)
        title = normalize_title(raw_title)
    elif not authors and before_year:
        # No detectable authors; title is everything before the year
        raw_title = before_year
        title = normalize_title(raw_title)
        try_before = True
    else:
        # IEEE-like: title is in before_year after stripping authors
        raw_title = _extract_title_before_year(before_year, authors)
        title = normalize_title(raw_title)
        try_before = True

    # Venue: remove extracted title from remaining text
    venue_text = ""
    if after_year and not try_before:
        raw_title = _extract_title_after_year(after_year).strip()
        remaining = after_year
        if raw_title and raw_title in remaining:
            remaining = remaining.replace(raw_title, "", 1)
        venue_text = remaining.strip().lstrip(".,; ")
    elif after_year:
        venue_text = after_year
    elif before_year and title:
        remaining = before_year[len(authors):].strip().lstrip(".,; ") if authors else before_year
        qm = re.search(r'"([^"]+)"', remaining)
        if qm:
            remaining = remaining.replace(qm.group(0), "", 1).strip().lstrip(".,; ")
        venue_text = remaining
    venue = _find_venue(venue_text, doi)

    return ParsedCitation(
        title=title,
        authors=authors,
        year=year,
        venue=venue,
        doi=doi,
    )
