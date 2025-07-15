from sqlalchemy import Column, Integer, String, Float, DateTime
from database import Base
import datetime

class AnalysisResult(Base):
    __tablename__ = "results"

    id = Column(Integer, primary_key=True, index=True)
    file_id = Column(String, unique=True, index=True)
    filename = Column(String)
    analysis_time = Column(DateTime, default=datetime.datetime.utcnow)
    avg_occupancy_rate = Column(Float)
    max_occupied_chairs = Column(Integer)
    total_interactions = Column(Integer)