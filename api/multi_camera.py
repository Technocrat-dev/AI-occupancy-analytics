"""
Multi-camera API endpoints for unified analytics across multiple camera feeds.
Uses database-backed configuration instead of in-memory global state.
"""
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from typing import List, Dict, Optional
import json
import logging

import database
import sql_models

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/multi-camera", tags=["multi-camera"])


# --- Pydantic Models ---

class CameraZone(BaseModel):
    """Defines a priority zone for a camera."""
    x1: int = Field(..., description="Top-left X coordinate")
    y1: int = Field(..., description="Top-left Y coordinate")
    x2: int = Field(..., description="Bottom-right X coordinate")
    y2: int = Field(..., description="Bottom-right Y coordinate")


class CameraConfigInput(BaseModel):
    """Configuration for a single camera (input)."""
    camera_id: int = Field(..., description="Unique camera identifier")
    priority: int = Field(default=1, ge=1, le=10, description="Camera priority (1-10, higher = more authoritative)")
    zones: List[CameraZone] = Field(default=[], description="Priority zones for this camera")
    video_path: Optional[str] = Field(None, description="Path to video file or stream URL")


class CameraConfigOutput(BaseModel):
    """Configuration for a single camera (output with DB id)."""
    id: int
    camera_id: int
    priority: int
    zones: List[CameraZone]
    video_path: Optional[str]

    class Config:
        from_attributes = True


class OverlapRegionInput(BaseModel):
    """Defines an overlapping region between cameras."""
    region: CameraZone = Field(..., description="The overlapping area")
    camera_ids: List[int] = Field(..., description="IDs of cameras that overlap in this region")


class MultiCameraSetupInput(BaseModel):
    """Complete multi-camera configuration (input)."""
    name: str = Field(..., description="Unique name for this setup")
    cameras: List[CameraConfigInput] = Field(..., description="List of camera configurations")
    overlap_regions: List[OverlapRegionInput] = Field(default=[], description="Overlapping regions between cameras")


class MultiCameraSetupOutput(BaseModel):
    """Complete multi-camera configuration (output)."""
    id: int
    name: str
    cameras: List[CameraConfigOutput]
    
    class Config:
        from_attributes = True


class UnifiedStats(BaseModel):
    """Unified occupancy statistics across all cameras."""
    total_chairs: int
    occupied_chairs: int
    occupancy_rate: float
    per_camera_stats: Dict[int, dict]


# --- Database Dependency ---

def get_db():
    db = database.SessionLocal()
    try:
        yield db
    finally:
        db.close()


# --- Endpoints ---

@router.post("/setup", response_model=dict)
async def configure_multi_camera(setup: MultiCameraSetupInput, db: Session = Depends(get_db)):
    """
    Configure multi-camera zones, priorities, and overlap regions.
    Creates or updates the configuration in the database.
    """
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
    
    # Check if setup with this name already exists
    existing = db.query(sql_models.MultiCameraSetup).filter(
        sql_models.MultiCameraSetup.name == setup.name
    ).first()
    
    if existing:
        # Delete old setup and recreate
        db.delete(existing)
        db.commit()
    
    # Create new setup
    db_setup = sql_models.MultiCameraSetup(name=setup.name)
    db.add(db_setup)
    db.flush()  # Get the ID
    
    # Add cameras
    for cam in setup.cameras:
        db_camera = sql_models.CameraConfig(
            setup_id=db_setup.id,
            camera_id=cam.camera_id,
            priority=cam.priority,
            video_path=cam.video_path,
            zones_json=json.dumps([z.model_dump() for z in cam.zones])
        )
        db.add(db_camera)
    
    # Add overlap regions
    for overlap in setup.overlap_regions:
        db_overlap = sql_models.OverlapRegion(
            setup_id=db_setup.id,
            region_json=json.dumps(overlap.region.model_dump()),
            camera_ids_json=json.dumps(overlap.camera_ids)
        )
        db.add(db_overlap)
    
    db.commit()
    
    logger.info(f"Multi-camera setup '{setup.name}' saved: {len(setup.cameras)} cameras, {len(setup.overlap_regions)} overlap regions")
    
    return {
        "success": True,
        "message": f"Configured {len(setup.cameras)} cameras",
        "setup_id": db_setup.id,
        "cameras": [{"id": c.camera_id, "priority": c.priority, "zones": len(c.zones)} for c in setup.cameras]
    }


@router.get("/setup", response_model=List[dict])
async def list_setups(db: Session = Depends(get_db)):
    """List all saved multi-camera setups."""
    setups = db.query(sql_models.MultiCameraSetup).all()
    return [
        {
            "id": s.id,
            "name": s.name,
            "camera_count": len(s.cameras),
            "created_at": s.created_at.isoformat() if s.created_at else None
        }
        for s in setups
    ]


@router.get("/setup/{setup_id}")
async def get_setup(setup_id: int, db: Session = Depends(get_db)):
    """Get a specific multi-camera configuration by ID."""
    setup = db.query(sql_models.MultiCameraSetup).filter(
        sql_models.MultiCameraSetup.id == setup_id
    ).first()
    
    if not setup:
        raise HTTPException(status_code=404, detail="Multi-camera setup not found")
    
    cameras = []
    for cam in setup.cameras:
        zones = json.loads(cam.zones_json) if cam.zones_json else []
        cameras.append({
            "camera_id": cam.camera_id,
            "priority": cam.priority,
            "video_path": cam.video_path,
            "zones": zones
        })
    
    # Get overlap regions
    overlaps = db.query(sql_models.OverlapRegion).filter(
        sql_models.OverlapRegion.setup_id == setup_id
    ).all()
    
    overlap_regions = []
    for o in overlaps:
        overlap_regions.append({
            "region": json.loads(o.region_json),
            "camera_ids": json.loads(o.camera_ids_json)
        })
    
    return {
        "id": setup.id,
        "name": setup.name,
        "cameras": cameras,
        "overlap_regions": overlap_regions,
        "created_at": setup.created_at.isoformat() if setup.created_at else None
    }


@router.delete("/setup/{setup_id}")
async def delete_setup(setup_id: int, db: Session = Depends(get_db)):
    """Delete a multi-camera configuration."""
    setup = db.query(sql_models.MultiCameraSetup).filter(
        sql_models.MultiCameraSetup.id == setup_id
    ).first()
    
    if not setup:
        raise HTTPException(status_code=404, detail="Multi-camera setup not found")
    
    db.delete(setup)
    db.commit()
    
    return {"success": True, "message": f"Setup '{setup.name}' deleted"}


@router.post("/process/{setup_id}")
async def process_multi_camera(setup_id: int, db: Session = Depends(get_db)):
    """
    Process multiple camera feeds with the specified configuration.
    
    Uses the refactored processor to analyze each camera's video and
    merge results using the MultiCameraManager.
    """
    from process_video import MultiCameraManager
    from services.processor import ProcessingSettings, process_frames
    from services.analytics import create_interaction_ledger, analyze_person_metrics, analyze_chair_metrics
    
    # Load setup from database
    setup = db.query(sql_models.MultiCameraSetup).filter(
        sql_models.MultiCameraSetup.id == setup_id
    ).first()
    
    if not setup:
        raise HTTPException(status_code=404, detail="Multi-camera setup not found")
    
    if not setup.cameras:
        raise HTTPException(status_code=400, detail="No cameras configured in this setup")
    
    # Initialize multi-camera manager
    multi_cam_manager = MultiCameraManager()
    
    # Configure zones and priorities
    for cam in setup.cameras:
        zones = json.loads(cam.zones_json) if cam.zones_json else []
        zone_tuples = [(z['x1'], z['y1'], z['x2'], z['y2']) for z in zones]
        multi_cam_manager.set_camera_zones(cam.camera_id, zone_tuples, priority=cam.priority)
    
    # Configure overlap regions
    overlaps = db.query(sql_models.OverlapRegion).filter(
        sql_models.OverlapRegion.setup_id == setup_id
    ).all()
    
    for o in overlaps:
        region = json.loads(o.region_json)
        camera_ids = json.loads(o.camera_ids_json)
        multi_cam_manager.add_overlap_region(
            [region['x1'], region['y1'], region['x2'], region['y2']],
            camera_ids
        )
    
    # Process each camera
    all_camera_results = {}
    per_camera_stats = {}
    errors = []
    
    for cam in setup.cameras:
        if not cam.video_path:
            logger.warning(f"Camera {cam.camera_id} has no video path, skipping")
            errors.append(f"Camera {cam.camera_id}: No video path specified")
            continue
        
        try:
            from pathlib import Path
            video_file = Path(cam.video_path)
            if not video_file.exists():
                error_msg = f"Camera {cam.camera_id}: Video file not found: {cam.video_path}"
                logger.error(error_msg)
                errors.append(error_msg)
                continue
                
            settings = ProcessingSettings()
            
            # Collect results from this camera
            camera_occupancy = []
            camera_chairs = {}
            
            for result in process_frames(cam.video_path, settings, annotate_frames=False):
                camera_chairs = {
                    chair_id: {
                        'center': chair_data['center'],
                        'bbox': chair_data['bbox'],
                        'is_occupied': result.chair_person_mapping.get(chair_id) is not None
                    }
                    for chair_id, chair_data in result.chairs.items()
                }
                camera_occupancy.append({
                    'frame': result.frame_number,
                    'occupied': result.occupied_chairs,
                    'total': result.total_chairs,
                    'rate': result.occupancy_rate
                })
            
            all_camera_results[cam.camera_id] = camera_chairs
            per_camera_stats[cam.camera_id] = {
                "total_frames": len(camera_occupancy),
                "avg_occupancy_rate": sum(o['rate'] for o in camera_occupancy) / len(camera_occupancy) if camera_occupancy else 0
            }
        except Exception as e:
            error_msg = f"Camera {cam.camera_id}: Processing error - {str(e)}"
            logger.error(error_msg)
            errors.append(error_msg)
    
    # Get unified stats using the manager
    unified = multi_cam_manager.get_unified_occupancy_count(all_camera_results)
    
    return {
        "success": len(errors) == 0 or len(all_camera_results) > 0,
        "setup_name": setup.name,
        "unified_stats": {
            "total_chairs": unified['total_chairs'],
            "occupied_chairs": unified['occupied_chairs'],
            "occupancy_rate": unified['occupancy_rate']
        },
        "per_camera_stats": per_camera_stats,
        "errors": errors,
        "cameras_processed": len(all_camera_results),
        "cameras_total": len(setup.cameras)
    }


@router.get("/stats/{setup_id}")
async def get_setup_stats(setup_id: int, db: Session = Depends(get_db)):
    """
    Get the most recent processing stats for a setup.
    Note: Currently returns empty as stats are not persisted between requests.
    For real-time stats, use the live streaming endpoints.
    """
    setup = db.query(sql_models.MultiCameraSetup).filter(
        sql_models.MultiCameraSetup.id == setup_id
    ).first()
    
    if not setup:
        raise HTTPException(status_code=404, detail="Multi-camera setup not found")
    
    return {
        "setup_id": setup_id,
        "setup_name": setup.name,
        "message": "Use POST /api/multi-camera/process/{setup_id} to get fresh stats"
    }
