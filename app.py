from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.encoders import jsonable_encoder
from typing import List, Optional
import schemas
import uuid
import shutil
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
from sqlalchemy.orm import Session

from config import settings
from process_video import process_video_for_api
import database, sql_models

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Chair Occupancy Tracker API",
    description=\"API for processing videos to track chair occupancy and provide detailed analytics.\",
    version=\"2.0.0\"
)

@app.get("/")
async def read_root():
    return FileResponse('index.html')

sql_models.Base.metadata.create_all(bind=database.engine)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include multi-camera API router
from api.multi_camera import router as multi_camera_router
app.include_router(multi_camera_router)

# Include WebSocket for real-time progress
from websocket_handler import include_websocket
include_websocket(app)

# Initialize directories and executor
settings.ensure_directories()
UPLOAD_DIR = settings.UPLOAD_DIR
OUTPUT_DIR = settings.OUTPUT_DIR

app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")

executor = ThreadPoolExecutor(max_workers=settings.MAX_WORKERS)

def get_db():
    db = database.SessionLocal()
    try:
        yield db
    finally:
        db.close()

def validate_video_file(filename: str, file_size: int) -> bool:
    """Validate uploaded video file by extension and size."""
    from pathlib import Path
    if file_size > settings.MAX_FILE_SIZE:
        return False
    return Path(filename).suffix.lower() in settings.ALLOWED_EXTENSIONS



@app.post("/process-video")
async def process_video_endpoint(
    file: UploadFile = File(...),
    proximity_threshold: Optional[int] = None,
    occupancy_frames_threshold: Optional[int] = None,
    motion_blur_threshold: Optional[int] = None,
    db: Session = Depends(get_db)
):
    """Process an uploaded video to track chair occupancy."""
    if not file.filename or not validate_video_file(file.filename, file.size or 0):
        raise HTTPException(status_code=400, detail="Invalid file format or size.")
    
    # Apply defaults and validate parameter ranges
    from pathlib import Path
    prox = settings.validate_proximity_threshold(
        proximity_threshold if proximity_threshold is not None else settings.DEFAULT_PROXIMITY_THRESHOLD
    )
    occ_frames = settings.validate_occupancy_frames(
        occupancy_frames_threshold if occupancy_frames_threshold is not None else settings.DEFAULT_OCCUPANCY_FRAMES
    )
    blur = settings.validate_motion_blur_threshold(
        motion_blur_threshold if motion_blur_threshold is not None else settings.DEFAULT_MOTION_BLUR_THRESHOLD
    )
    
    unique_id = str(uuid.uuid4())
    input_path = UPLOAD_DIR / f"{unique_id}{Path(file.filename).suffix}"
    output_path = OUTPUT_DIR / f"{unique_id}_output.mp4"
    results_path = OUTPUT_DIR / f"{unique_id}_results.json"
    
    try:
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        processing_settings = {
            'proximity_threshold': prox,
            'occupancy_frames_threshold': occ_frames,
            'motion_blur_threshold': blur
        }
        
        loop = asyncio.get_event_loop()
        processing_result = await loop.run_in_executor(
            executor, process_video_for_api, str(input_path), str(output_path), processing_settings
        )
        
        serializable_results = jsonable_encoder(processing_result)
        
        with open(results_path, "w") as f:
            json.dump(serializable_results, f, indent=4)
            
        new_result_entry = sql_models.AnalysisResult(
            file_id=unique_id,
            filename=file.filename,
            avg_occupancy_rate=serializable_results.get('average_occupancy_rate', 0),
            max_occupied_chairs=serializable_results.get('max_occupied_chairs', 0),
            total_interactions=len(serializable_results.get('interaction_ledger', []))
        )
        db.add(new_result_entry)
        db.commit()
        db.refresh(new_result_entry)
        logger.info(f"Saved analysis summary for file_id {unique_id} to database.")
        
        response_data = {
            "success": True,
            "message": "Video processed successfully. Results saved and available via API.",
            "file_id": unique_id,
            "output_video_url": f"/outputs/{unique_id}_output.mp4",
            "results_api_url": f"/api/results/{unique_id}",
            "processing_results": serializable_results,
        }
        
        if input_path.exists():
            input_path.unlink()
        
        return JSONResponse(content=response_data)
        
    except Exception as e:
        logger.error(f"Error during video processing: {str(e)}")
        for path in [input_path, output_path, results_path]:
            if path.exists():
                path.unlink()
        raise HTTPException(status_code=500, detail=f"Error processing video: {str(e)}")

@app.get("/api/results/{file_id}")
async def get_analysis_results(file_id: str):
    results_path = OUTPUT_DIR / f"{file_id}_results.json"
    if not results_path.is_file():
        raise HTTPException(status_code=404, detail="Analysis results not found.")
    with open(results_path, "r") as f:
        return JSONResponse(content=json.load(f))

@app.get("/api/history", response_model=List[schemas.AnalysisResult])
async def get_analysis_history(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    """
    Retrieve a list of all past analysis summaries from the database.
    """
    results = db.query(sql_models.AnalysisResult).order_by(sql_models.AnalysisResult.analysis_time.desc()).offset(skip).limit(limit).all()
    return results

@app.get("/download/{file_id}")
async def download_processed_video(file_id: str):
    video_path = OUTPUT_DIR / f"{file_id}_output.mp4"
    if not video_path.is_file():
        raise HTTPException(status_code=404, detail="Processed video not found.")
    return FileResponse(path=video_path, filename=f"processed_{file_id}.mp4", media_type="video/mp4")

@app.delete("/api/history/{file_id}")
async def delete_analysis_history(file_id: str, db: Session = Depends(get_db)):
    """
    Delete an analysis history record and its associated files.
    """
    # Find the database record
    result = db.query(sql_models.AnalysisResult).filter(sql_models.AnalysisResult.file_id == file_id).first()
    
    if not result:
        raise HTTPException(status_code=404, detail="Analysis record not found.")
    
    # Define paths for the associated files
    output_video_path = OUTPUT_DIR / f"{file_id}_output.mp4"
    results_json_path = OUTPUT_DIR / f"{file_id}_results.json"
    
    # Delete the files from the server if they exist
    try:
        if output_video_path.exists():
            output_video_path.unlink()
            logger.info(f"Deleted output video: {output_video_path}")
        if results_json_path.exists():
            results_json_path.unlink()
            logger.info(f"Deleted results JSON: {results_json_path}")
    except Exception as e:
        logger.error(f"Error deleting files for file_id {file_id}: {str(e)}")
        # Decide if you want to stop or continue even if file deletion fails
        # For now, we'll continue to delete the DB record
    
    # Delete the record from the database
    db.delete(result)
    db.commit()
    
    logger.info(f"Successfully deleted analysis record for file_id: {file_id}")
    
    return JSONResponse(content={"success": True, "message": f"Analysis {file_id} deleted successfully."})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)