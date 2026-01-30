"""
Frame processing generator for chair occupancy detection.
Provides a reusable generator that can be consumed by both file processing and live streaming.
"""
import cv2
import time
from typing import Dict, Any, Generator, Optional, Callable, Union
from dataclasses import dataclass

from myutils.detector import YOLODetector
from myutils.tracker import convert_to_deepsort_format
from deep_sort_realtime.deepsort_tracker import DeepSort


@dataclass
class ProcessingSettings:
    """Configuration for video processing."""
    proximity_threshold: int = 80
    occupancy_frames_threshold: int = 5
    motion_blur_threshold: int = 100
    model_path: str = "yolo11m.pt"
    device: str = None


@dataclass
class FrameResult:
    """Result from processing a single frame."""
    frame: Any  # numpy array
    frame_number: int
    total_chairs: int
    occupied_chairs: int
    occupancy_rate: float
    chair_person_mapping: Dict
    chairs: Dict
    is_blurry: bool = False


def create_components(settings: ProcessingSettings):
    """Initialize detection and tracking components."""
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if settings.device:
        device = settings.device
    
    print(f"[INFO] Using device: {device} (CUDA available: {torch.cuda.is_available()})")
    
    # Import here to avoid circular imports
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from process_video import ChairOccupancyTracker
    
    detector = YOLODetector(model_path=settings.model_path, device=device)
    tracker = DeepSort(max_age=50, n_init=2, nms_max_overlap=1.0, max_cosine_distance=0.4)
    occupancy_tracker = ChairOccupancyTracker(
        proximity_threshold=settings.proximity_threshold,
        occupancy_frames_threshold=settings.occupancy_frames_threshold,
        motion_blur_threshold=settings.motion_blur_threshold
    )
    
    return detector, tracker, occupancy_tracker


def process_frames(
    video_source: Union[str, int],
    settings: Optional[ProcessingSettings] = None,
    on_frame_callback: Optional[Callable[[FrameResult], None]] = None,
    annotate_frames: bool = True
) -> Generator[FrameResult, None, Dict[str, Any]]:
    """
    Generator that processes video frames and yields results.
    
    Args:
        video_source: Path to video file, camera index (0, 1, etc.), or RTSP URL
        settings: Processing configuration
        on_frame_callback: Optional callback invoked after each frame
        annotate_frames: Whether to draw bounding boxes on frames
        
    Yields:
        FrameResult for each processed frame
        
    Returns:
        Final processing statistics (accessible via generator.value after StopIteration)
    """
    if settings is None:
        settings = ProcessingSettings()
    
    detector, tracker, occupancy_tracker = create_components(settings)
    
    cap = cv2.VideoCapture(video_source)
    
    # Check if camera opened successfully
    if not cap.isOpened():
        error_msg = f"Failed to open video source: {video_source}. "
        if isinstance(video_source, int):
            error_msg += "Please check if the webcam is connected and not being used by another application."
        else:
            error_msg += "Please verify the video file path or RTSP URL is correct."
        raise RuntimeError(error_msg)
    
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    frame_count = 0
    occupancy_history = []
    person_chair_map_history = []
    max_occupied = 0
    max_chairs = 0
    start_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Brightness change detection and enhancement
            brightness_changed, brightness_diff = occupancy_tracker.detect_brightness_change(frame)
            if brightness_changed:
                frame = occupancy_tracker.enhance_frame_adaptive(frame, brightness_diff)
            
            # Motion blur detection
            is_blurry = occupancy_tracker.detect_motion_blur(frame)
            if is_blurry:
                frame_count += 1
                continue
            
            # Reflection detection
            occupancy_tracker.detect_reflections(frame)
            
            # Detection
            detections = detector.detect(frame)
            person_detections = detections[detections['class'] == 0]
            chair_detections = detections[detections['class'] == 56]
            
            # Tracking
            deepsort_detections = convert_to_deepsort_format(person_detections)
            tracks = tracker.update_tracks(deepsort_detections, frame=frame)
            
            # Occupancy analysis
            chairs = occupancy_tracker.assign_chair_ids(chair_detections, frame)
            frame_occupancy = occupancy_tracker.update_occupancy(chairs, tracks)
            
            total_chairs = len(chairs)
            occupied_chairs = sum(1 for cid in chairs if occupancy_tracker.get_occupancy_status(cid))
            occupancy_rate = occupied_chairs / total_chairs if total_chairs > 0 else 0
            
            max_occupied = max(max_occupied, occupied_chairs)
            max_chairs = max(max_chairs, total_chairs)
            
            # Annotate frame if requested
            if annotate_frames:
                frame = _annotate_frame(frame, chairs, tracks, occupancy_tracker,
                                       total_chairs, occupied_chairs, occupancy_rate)
            
            # Build result
            result = FrameResult(
                frame=frame,
                frame_number=frame_count,
                total_chairs=total_chairs,
                occupied_chairs=occupied_chairs,
                occupancy_rate=occupancy_rate,
                chair_person_mapping=occupancy_tracker.chair_person_mapping.copy(),
                chairs=chairs
            )
            
            # Track history for analytics
            occupancy_history.append({'frame': frame_count, 'occupancy_rate': occupancy_rate})
            person_chair_map_history.append({'frame': frame_count, 'mapping': result.chair_person_mapping})
            
            if on_frame_callback:
                on_frame_callback(result)
            
            yield result
            
            frame_count += 1
            
    finally:
        cap.release()
        print(f"[INFO] Camera released for source: {video_source}")
    
    # Return final statistics (accessible via generator.value)
    processing_time = time.time() - start_time
    avg_occupancy = sum(h['occupancy_rate'] for h in occupancy_history) / len(occupancy_history) if occupancy_history else 0
    
    return {
        'total_frames_processed': frame_count,
        'processing_time_seconds': processing_time,
        'fps': fps,
        'total_chairs_detected': max_chairs,
        'max_occupied_chairs': max_occupied,
        'average_occupancy_rate': avg_occupancy,
        'occupancy_history': occupancy_history,
        'person_chair_map_history': person_chair_map_history
    }


def _annotate_frame(frame, chairs, tracks, occupancy_tracker, total_chairs, occupied_chairs, occupancy_rate):
    """Draw bounding boxes and labels on frame."""
    # Draw chairs
    for chair_id, chair_data in chairs.items():
        bbox = chair_data['bbox']
        x1, y1, x2, y2 = map(int, bbox)
        is_occupied = occupancy_tracker.get_occupancy_status(chair_id)
        
        chair_color = (0, 0, 255)  # Red = Empty
        if is_occupied:
            chair_color = (0, 255, 0)  # Green = Occupied
        if chair_data.get('interpolated', False):
            chair_color = (255, 165, 0)  # Orange = Interpolated
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), chair_color, 2)
        label = f"Chair {chair_id}"
        if is_occupied:
            person_id = occupancy_tracker.chair_person_mapping.get(chair_id, ['?'])[0]
            label += f" (P:{person_id})"
        cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, chair_color, 2)
    
    # Draw people
    for track in tracks:
        if not track.is_confirmed():
            continue
        bbox = track.to_ltrb()
        track_id = track.track_id
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, f"Person {track_id}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    # Draw stats overlay
    stats_text = [f"Chairs: {total_chairs}", f"Occupied: {occupied_chairs}", f"Rate: {occupancy_rate:.1%}"]
    for i, text in enumerate(stats_text):
        cv2.putText(frame, text, (10, 30 + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return frame
