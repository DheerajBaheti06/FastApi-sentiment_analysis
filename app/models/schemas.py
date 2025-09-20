from typing import List, Optional
from pydantic import BaseModel, Field

"""Schemas for compact analyze endpoint only."""


class AnalyzeCompactRequest(BaseModel):
    texts: List[str]
    max_summary_sentences: int = Field(3, ge=1, le=10)
    include_wordcloud: bool = Field(True)


class WordcloudRef(BaseModel):
    url: str
    key: str


class AnalyzeCompactResponse(BaseModel):
    sentiments: List[str]
    summary: str
    percentages: dict | None = None  # keys: positive, negative, neutral (values 0-100)
    wordcloud: Optional[WordcloudRef] = None


class PredictRequest(BaseModel):
    texts: List[str]


class PredictResponse(BaseModel):
    sentiments: List[str]
