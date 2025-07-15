from pydantic import BaseModel
import datetime
from typing import Optional

class AnalysisResult(BaseModel):
    id: int
    file_id: str
    filename: str
    analysis_time: datetime.datetime
    avg_occupancy_rate: float
    max_occupied_chairs: int
    total_interactions: int

    class Config:
        from_attributes = True