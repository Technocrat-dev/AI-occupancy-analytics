"""
MJPEG streaming service for live video feed.
Provides real-time annotated video stream over HTTP.
"""
import cv2
from typing import Generator, Optional, Union

from services.processor import ProcessingSettings, process_frames


def generate_mjpeg_stream(
    video_source: Union[str, int],
    settings: Optional[ProcessingSettings] = None,
    jpeg_quality: int = 80
) -> Generator[bytes, None, None]:
    """
    Generator that yields MJPEG frames for HTTP streaming.
    
    Args:
        video_source: Camera index (0, 1, etc.) or RTSP URL string
        settings: Processing configuration
        jpeg_quality: JPEG encoding quality (0-100)
        
    Yields:
        MJPEG frame bytes suitable for multipart/x-mixed-replace response
    """
    if settings is None:
        settings = ProcessingSettings()
    
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
    
    frame_count = 0
    try:
        print(f"[INFO] Starting MJPEG stream for source: {video_source}")
        for result in process_frames(video_source, settings, annotate_frames=True):
            # Encode frame as JPEG
            success, jpeg = cv2.imencode('.jpg', result.frame, encode_params)
            if not success:
                print(f"[WARN] Failed to encode frame {frame_count}")
                continue
            
            # Yield as MJPEG multipart format
            yield (
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n'
            )
            
            frame_count += 1
            if frame_count == 1:
                print(f"[INFO] First frame yielded successfully")
                
    except RuntimeError as e:
        print(f"[ERROR] Stream error: {e}")
        # Create error frame
        import numpy as np
        error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(error_frame, "Camera Error", (50, 200), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        cv2.putText(error_frame, str(e)[:60], (50, 250), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        success, jpeg = cv2.imencode('.jpg', error_frame, encode_params)
        if success:
            yield (
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n'
            )
        raise  # Re-raise to stop the stream
    except GeneratorExit:
        print(f"[INFO] Stream client disconnected after {frame_count} frames")
        raise
    finally:
        print(f"[INFO] MJPEG stream ended for source: {video_source}")


def generate_stats_stream(
    video_source: Union[str, int],
    settings: Optional[ProcessingSettings] = None,
    interval_frames: int = 10
) -> Generator[dict, None, None]:
    """
    Generator that yields occupancy stats at regular intervals.
    Can be used with WebSocket to push real-time updates.
    
    Args:
        video_source: Camera index or RTSP URL
        settings: Processing configuration
        interval_frames: How often to yield stats (every N frames)
        
    Yields:
        Dict with current occupancy stats
    """
    if settings is None:
        settings = ProcessingSettings()
    
    for result in process_frames(video_source, settings, annotate_frames=False):
        if result.frame_number % interval_frames == 0:
            yield {
                "frame": result.frame_number,
                "total_chairs": result.total_chairs,
                "occupied_chairs": result.occupied_chairs,
                "occupancy_rate": result.occupancy_rate,
                "chair_person_mapping": {
                    str(k): v for k, v in result.chair_person_mapping.items()
                }
            }
