from pydantic import BaseModel, Field
from typing import List


class SummaryMeetingAndKeywordMeetingByLLM(BaseModel):
    summary_meeting: str = Field(
        ..., 
        description="Gemini(LLM)가 원본 회의록을 요약한 핵심 요약본"
    )
    keyword_meeting_list: List[str] = Field(
        ..., 
        description="회의록의 핵심 내용을 대표하는 키워드 리스트 (정확히 5개). 예: ['AI', '신제품', '시장 동향']"
    )


class SummaryNewsByLLM(BaseModel):
    summary: str = Field(
        ...,
        description="Gemini(LLM)가 원본 뉴스를 요약한 핵심 요약본"
    )
