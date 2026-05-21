from pydantic import BaseModel
from typing import Optional


class ParsedCitation(BaseModel):
    title: str = ""
    authors: str = ""
    year: Optional[int] = None
    venue: str = ""
    doi: str = ""
