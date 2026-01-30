from sqlalchemy import Column, Integer, String, Float, DateTime, Text, ForeignKey
from sqlalchemy.orm import relationship
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


class MultiCameraSetup(Base):
    """Stores multi-camera configuration setups."""
    __tablename__ = "multi_camera_setups"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)
    
    # Relationship to cameras
    cameras = relationship("CameraConfig", back_populates="setup", cascade="all, delete-orphan")


class CameraConfig(Base):
    """Stores individual camera configuration within a multi-camera setup."""
    __tablename__ = "camera_configs"

    id = Column(Integer, primary_key=True, index=True)
    setup_id = Column(Integer, ForeignKey("multi_camera_setups.id", ondelete="CASCADE"))
    camera_id = Column(Integer)  # User-defined camera identifier
    priority = Column(Integer, default=1)
    video_path = Column(String, nullable=True)  # Path to video file or RTSP URL
    zones_json = Column(Text, nullable=True)  # Store zones as JSON string
    
    # Relationship back to setup
    setup = relationship("MultiCameraSetup", back_populates="cameras")


class OverlapRegion(Base):
    """Stores overlap regions between cameras."""
    __tablename__ = "overlap_regions"

    id = Column(Integer, primary_key=True, index=True)
    setup_id = Column(Integer, ForeignKey("multi_camera_setups.id", ondelete="CASCADE"))
    region_json = Column(Text)  # Store region coords as JSON: {"x1": ..., "y1": ..., "x2": ..., "y2": ...}
    camera_ids_json = Column(Text)  # Store as JSON array: [1, 2]
