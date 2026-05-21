from typing import Literal

from pydantic import BaseModel


class LLMOutput(BaseModel):
    label: Literal["VALID", "PARTIALLY_VALID", "HALLUCINATED"]
    confidence: float
    reason: str
