"""
Multi-camera API endpoints for unified analytics across multiple camera feeds.
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/multi-camera", tags=["multi-camera"])


# --- Pydantic Models ---

class CameraZone(BaseModel):
    """Defines a priority zone for a camera."""
    x1: int = Field(..., description="Top-left X coordinate")
    y1: int = Field(..., description="Top-left Y coordinate")
    x2: int = Field(..., description="Bottom-right X coordinate")
    y2: int = Field(..., description="Bottom-right Y coordinate")


class CameraConfig(BaseModel):
    """Configuration for a single camera."""
    camera_id: int = Field(..., description="Unique camera identifier")
    priority: int = Field(default=1, ge=1, le=10, description="Camera priority (1-10, higher = more authoritative)")
    zones: List[CameraZone] = Field(default=[], description="Priority zones for this camera")
    video_path: Optional[str] = Field(None, description="Path to video file or stream URL")


class OverlapRegion(BaseModel):
    """Defines an overlapping region between cameras."""
    region: CameraZone = Field(..., description="The overlapping area")
    camera_ids: List[int] = Field(..., description="IDs of cameras that overlap in this region")


class MultiCameraSetup(BaseModel):
    """Complete multi-camera configuration."""
    cameras: List[CameraConfig] = Field(..., description="List of camera configurations")
    overlap_regions: List[OverlapRegion] = Field(default=[], description="Overlapping regions between cameras")


class UnifiedStats(BaseModel):
    """Unified occupancy statistics across all cameras."""
    total_chairs: int
    occupied_chairs: int
    occupancy_rate: float
    per_camera_stats: Dict[int, dict]


# --- In-Memory State (for demo; use Redis/DB in production) ---

_current_setup: Optional[MultiCameraSetup] = None
_unified_stats: Optional[UnifiedStats] = None


# --- Endpoints ---

@router.post("/setup", response_model=dict)
async def configure_multi_camera(setup: MultiCameraSetup):
    """
    Configure multi-camera zones, priorities, and overlap regions.
    This must be called before processing multiple camera feeds.
    """
    global _current_setup
    
    # Validate camera IDs are unique
    camera_ids = [cam.camera_id for cam in setup.cameras]
    if len(camera_ids) != len(set(camera_ids)):
        raise HTTPException(status_code=400, detail="Duplicate camera IDs found")
    
    # Validate overlap regions reference existing cameras
    for overlap in setup.overlap_regions:
        for cam_id in overlap.camera_ids:
            if cam_id not in camera_ids:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Overlap region references unknown camera_id: {cam_id}"
                )
    
    _current_setup = setup
    
    logger.info(f"Multi-camera setup configured: {len(setup.cameras)} cameras, {len(setup.overlap_regions)} overlap regions")
    
    return {
        "success": True,
        "message": f"Configured {len(setup.cameras)} cameras",
        "cameras": [{"id": c.camera_id, "priority": c.priority, "zones": len(c.zones)} for c in setup.cameras]
    }


@router.get("/setup", response_model=Optional[MultiCameraSetup])
async def get_current_setup():
    """Get the current multi-camera configuration."""
    if _current_setup is None:
        raise HTTPException(status_code=404, detail="No multi-camera setup configured")
    return _current_setup


@router.delete("/setup")
async def clear_setup():
    """Clear the current multi-camera configuration."""
    global _current_setup, _unified_stats
    _current_setup = None
    _unified_stats = None
    return {"success": True, "message": "Multi-camera setup cleared"}


@router.get("/stats", response_model=Optional[UnifiedStats])
async def get_unified_stats():
    """
    Get unified occupancy statistics across all configured cameras.
    Note: This returns cached stats from the last processing run.
    """
    if _unified_stats is None:
        raise HTTPException(
            status_code=404, 
            detail="No unified stats available. Process videos first."
        )
    return _unified_stats


@router.post("/process")
async def process_multi_camera():
    """
    Process multiple camera feeds with the current configuration.
    
    Note: This is a placeholder for the full implementation.
    Full implementation would:
    1. Load videos from each camera's video_path
    2. Run detection and tracking in parallel
    3. Apply zone priorities and overlap resolution
    4. Generate unified analytics
    """
    if _current_setup is None:
        raise HTTPException(
            status_code=400, 
            detail="No multi-camera setup configured. Call POST /api/multi-camera/setup first."
        )
    
    # Placeholder response - actual implementation would process videos
    return {
        "success": True,
        "message": "Multi-camera processing is configured but not yet implemented for API mode",
        "configured_cameras": len(_current_setup.cameras),
        "hint": "Use the standalone process_video.py with USE_MULTI_CAMERA=True for now"
    }


# --- Helper to include router in main app ---

def include_router(app):
    """Include the multi-camera router in a FastAPI app."""
    app.include_router(router)
