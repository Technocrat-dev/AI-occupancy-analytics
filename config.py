"""
Configuration management with environment variable support for Docker deployment.
"""
import os
from pathlib import Path


class Settings:
    """Application settings loaded from environment variables with sensible defaults."""
    
    # Paths
    UPLOAD_DIR: Path = Path(os.getenv("UPLOAD_DIR", "uploads"))
    OUTPUT_DIR: Path = Path(os.getenv("OUTPUT_DIR", "outputs"))
    MODEL_PATH: str = os.getenv("MODEL_PATH", "yolo11m.pt")
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./analysis.db")
    
    # Processing defaults
    DEFAULT_PROXIMITY_THRESHOLD: int = int(os.getenv("DEFAULT_PROXIMITY_THRESHOLD", "80"))
    DEFAULT_OCCUPANCY_FRAMES: int = int(os.getenv("DEFAULT_OCCUPANCY_FRAMES", "5"))
    DEFAULT_MOTION_BLUR_THRESHOLD: int = int(os.getenv("DEFAULT_MOTION_BLUR_THRESHOLD", "100"))
    
    # Limits
    MAX_FILE_SIZE: int = int(os.getenv("MAX_FILE_SIZE", str(100 * 1024 * 1024)))  # 100MB
    MAX_WORKERS: int = int(os.getenv("MAX_WORKERS", "2"))
    
    # Allowed video extensions
    ALLOWED_EXTENSIONS: set = {".mp4", ".avi", ".mov", ".mkv", ".wmv"}
    
    # Validation ranges for ML parameters
    PROXIMITY_THRESHOLD_RANGE: tuple = (10, 500)
    OCCUPANCY_FRAMES_RANGE: tuple = (1, 60)
    MOTION_BLUR_THRESHOLD_RANGE: tuple = (10, 500)
    
    @classmethod
    def validate_proximity_threshold(cls, value: int) -> int:
        """Validate and clamp proximity threshold to valid range."""
        min_val, max_val = cls.PROXIMITY_THRESHOLD_RANGE
        return max(min_val, min(max_val, value))
    
    @classmethod
    def validate_occupancy_frames(cls, value: int) -> int:
        """Validate and clamp occupancy frames to valid range."""
        min_val, max_val = cls.OCCUPANCY_FRAMES_RANGE
        return max(min_val, min(max_val, value))
    
    @classmethod
    def validate_motion_blur_threshold(cls, value: int) -> int:
        """Validate and clamp motion blur threshold to valid range."""
        min_val, max_val = cls.MOTION_BLUR_THRESHOLD_RANGE
        return max(min_val, min(max_val, value))
    
    @classmethod
    def ensure_directories(cls):
        """Create required directories if they don't exist."""
        cls.UPLOAD_DIR.mkdir(exist_ok=True)
        cls.OUTPUT_DIR.mkdir(exist_ok=True)


# Singleton instance for easy import
settings = Settings()
