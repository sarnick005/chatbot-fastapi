from pydantic import BaseModel
from typing import List


class Pattern(BaseModel):
    pattern: str


class Response(BaseModel):
    response: str


class Intent(BaseModel):
    tag: str
    patterns: List[str]
    responses: List[str]


class IntentsData(BaseModel):
    intents: List[Intent]


class ImportSummary(BaseModel):
    status: str
    new_intents_count: int
    new_patterns_count: int
    new_responses_count: int
    updated_intents_count: int
    message: str
