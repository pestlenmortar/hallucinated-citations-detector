import re

from .models import ParsedCitation
from .normalization import normalize_title

YEAR_RE = r"\b((?:1[89]\d{2}|20[0-2]\d))\b"

DOI_RE = (
    r"(?:https?://(?:dx\.)?doi\.org/|doi:?\s*)?"
    r"(10\.\d{4,}(?:\.\d+)*/[^\s,;)\]}>]+)"
)

# Single IEEE-style author name.
# Matches: A. Vaswani, A. N. Gomez, M.-W. Chang, H. van Hasselt,
#          G. van den Driessche, Y. LeCun, L. Fei-Fei, W. tau Yih,
#          J. Pouget-Abadie, J. Ba, Q. V. Le, L. Kaiser
IEE_AUTHOR = (
    r"[A-Z\u0141]\."                    # A. (including \u0141 = L-with-stroke)
    r"(?:[\s-]*[A-Z\u0141]\.)*"         # optional more initials (space or hyphen separated)
    r"\s+"                               # space before prefixes / last name
    r"(?:[a-z]+\s+)*"                   # optional lowercase prefixes (van, de, den, tau, ...)
    r"[A-Z][a-zA-Z]+"                   # last name (camelCase friendly)
    r"(?:-[A-Z][a-zA-Z]+)*"             # optional hyphenated extensions (-Badie, -Fei)
)

# Full IEEE author list with comma / and separators and optional "et al."
AUTHOR_IEE_RE = (
    IEE_AUTHOR +
    r"(?:" +
        # separator between authors: ", A. Smith"  or  ", and A. Smith"  or  " and A. Smith"
        r"(?:,\s*(?:and\s+)?|\s+and\s+)" +
        IEE_AUTHOR +
    r")*" +
    r"(?:\s+et\s+al\.)?" +
    r"\s*"
)

# Fallback for single-word institutional authors like "OpenAI"
INSTITUTIONAL_AUTHOR_RE = r"([A-Z][a-zA-Z]+(?:[A-Z][a-z]*)*)\s*,\s*\""


def _find_doi(text: str) -> str:
    m = re.search(DOI_RE, text, re.IGNORECASE)
    return m.group(1).rstrip(".,;") if m else ""


def _find_year(text: str) -> int | None:
    m = re.search(YEAR_RE, text)
    return int(m.group(1)) if m else None


def _find_authors(text: str) -> str:
    m = re.match(AUTHOR_IEE_RE, text)
    if m:
        authors = m.group(0).strip()
        authors = re.sub(r"\s+et\s+al\.?$", "", authors)
        return authors.rstrip(".,; ")
    # Fallback: institutional author like "OpenAI"
    m = re.match(INSTITUTIONAL_AUTHOR_RE, text)
    if m:
        return m.group(1).strip(".,; ")
    return ""


def _find_venue(text: str, doi: str) -> str:
    if doi:
        idx = text.lower().find(doi.lower())
        if idx != -1:
            text = text[:idx]
    text = re.sub(r"\b\d+\s*\(\d+\)\s*:?\s*\d+[-–]\d+\b", "", text)
    text = re.sub(r"(?:vol|no|pp)\.?\s*\d+[^\s,;]*", "", text, flags=re.IGNORECASE)
    text = re.sub(r",\s*\d{4}\b", "", text)
    text = re.sub(r"[;:,.\s]+$", "", text.strip())
    text = re.sub(r",\s*,", ",", text)
    text = re.sub(r",\s*,", ",", text)
    text = text.strip().rstrip(".,; ")
    return text


def ieee_last_names(author_str: str) -> set[str]:
    """Extract lowercased last names from IEEE author strings.
    Removes initials (single letter + period) and takes the last
    remaining token per author as the last name.
    Returns empty set for empty input.
    """
    if not author_str:
        return set()
    names = set()
    for part in re.split(r",\s*(?:and\s+)?|\s+and\s+", author_str):
        part = part.strip().rstrip(".,;")
        if not part or re.match(r"et\s+al\.?$", part, re.IGNORECASE):
            continue
        tokens = part.lower().split()
        non_initials = [t for t in tokens if not re.match(r"^[a-z]\.$", t)]
        if non_initials:
            names.add(non_initials[-1])
    return names


def db_last_names(author_str: str) -> set[str]:
    """Extract lowercased last names from DB-style full author strings.
    Takes the last token of each comma-separated name part.
    """
    if not author_str:
        return set()
    names = set()
    for part in re.split(r",\s*(?:and\s+)?|\s+and\s+", author_str):
        part = part.strip().rstrip(".,;")
        if not part:
            continue
        tokens = part.split()
        if tokens:
            names.add(tokens[-1].lower())
    return names


def ieee_author_overlap(ieee_authors: str, db_authors: str) -> float:
    """Compute Jaccard overlap using last-name matching.
    Works for IEEE-initials format vs DB full-name format.
    """
    ieee_names = ieee_last_names(ieee_authors)
    db_names = db_last_names(db_authors)
    if not ieee_names or not db_names:
        return 0.0
    return len(ieee_names & db_names) / len(ieee_names | db_names)


def parse_citation(raw: str) -> ParsedCitation:
    text = raw.strip()
    doi = _find_doi(text)
    year = _find_year(text)

    # Strip DOI from text for further processing
    text_no_doi = re.sub(DOI_RE, "", text, flags=re.IGNORECASE).strip()

    before_year = text_no_doi
    after_year = ""
    if year is not None:
        ypos = text_no_doi.find(str(year))
        if ypos != -1:
            before_year = text_no_doi[:ypos].strip().rstrip(".,; (")
            after_year = text_no_doi[ypos + 4:].strip().lstrip(".,;: )")

    authors = _find_authors(before_year)

    # Extract title — IEEE format uses double quotes
    title = ""
    quoted_title = re.search(r'"([^"]+)"', text_no_doi)
    if quoted_title:
        title = normalize_title(quoted_title.group(1))
    else:
        # Fallback to before_year-based title extraction (APA / title-first)
        remaining = before_year
        if authors and remaining.startswith(authors):
            remaining = remaining[len(authors):].strip().lstrip(".,; ")
        if remaining:
            title = normalize_title(remaining)

    # Extract venue — in IEEE it comes after ", in " following the title
    venue_text = ""
    if quoted_title:
        after_quote = text_no_doi[quoted_title.end():].strip().lstrip(".,; ")
        if after_quote.lower().startswith("in "):
            after_quote = after_quote[3:].strip()
        venue_text = after_quote
    elif before_year:
        remaining = before_year
        if authors and remaining.startswith(authors):
            remaining = remaining[len(authors):].strip().lstrip(".,; ")
        venue_text = remaining

    venue = _find_venue(venue_text, doi)

    return ParsedCitation(
        title=title,
        authors=authors,
        year=year,
        venue=venue,
        doi=doi,
    )
