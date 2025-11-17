from pydantic import BaseModel
from datetime import datetime
from typing import List, Optional

class MeetingData(BaseModel):
    id: int
    title: str
    created_dt: datetime

class MeetingDataByID(MeetingData):
    original_meeting: str
    summary_meeting: str
    # V3 변경사항
    news_items: Optional[List] = None
    keywords: Optional[List] = None
