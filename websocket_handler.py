"""
WebSocket handler for real-time video processing progress updates.
"""
from fastapi import WebSocket, WebSocketDisconnect
from typing import Dict, Set
import asyncio
import json
import logging

logger = logging.getLogger(__name__)


class ProgressManager:
    """Manages WebSocket connections for processing progress updates."""
    
    def __init__(self):
        # Map of file_id -> set of connected websockets
        self.active_connections: Dict[str, Set[WebSocket]] = {}
        # Map of file_id -> current progress data
        self.progress_data: Dict[str, dict] = {}
    
    async def connect(self, websocket: WebSocket, file_id: str):
        """Accept a new WebSocket connection for a file_id."""
        await websocket.accept()
        
        if file_id not in self.active_connections:
            self.active_connections[file_id] = set()
        
        self.active_connections[file_id].add(websocket)
        logger.info(f"WebSocket connected for file_id: {file_id}")
        
        # Send current progress if available
        if file_id in self.progress_data:
            await websocket.send_json(self.progress_data[file_id])
    
    def disconnect(self, websocket: WebSocket, file_id: str):
        """Remove a WebSocket connection."""
        if file_id in self.active_connections:
            self.active_connections[file_id].discard(websocket)
            if not self.active_connections[file_id]:
                del self.active_connections[file_id]
        logger.info(f"WebSocket disconnected for file_id: {file_id}")
    
    async def broadcast_progress(self, file_id: str, progress: dict):
        """Broadcast progress update to all connected clients for a file_id."""
        self.progress_data[file_id] = progress
        
        if file_id not in self.active_connections:
            return
        
        disconnected = set()
        for websocket in self.active_connections[file_id]:
            try:
                await websocket.send_json(progress)
            except Exception:
                disconnected.add(websocket)
        
        # Clean up disconnected clients
        for ws in disconnected:
            self.active_connections[file_id].discard(ws)
    
    def update_progress(self, file_id: str, frame: int, total_frames: int, 
                        occupied: int = 0, total_chairs: int = 0):
        """
        Update progress (non-async version for use in sync processing code).
        Call this from the video processing loop.
        """
        progress = {
            "file_id": file_id,
            "status": "processing",
            "frame": frame,
            "total_frames": total_frames,
            "percent": round((frame / total_frames) * 100, 1) if total_frames > 0 else 0,
            "occupied_chairs": occupied,
            "total_chairs": total_chairs
        }
        self.progress_data[file_id] = progress
        return progress
    
    def complete(self, file_id: str, success: bool = True, message: str = ""):
        """Mark processing as complete."""
        progress = {
            "file_id": file_id,
            "status": "complete" if success else "error",
            "percent": 100 if success else self.progress_data.get(file_id, {}).get("percent", 0),
            "message": message
        }
        self.progress_data[file_id] = progress
        return progress
    
    def cleanup(self, file_id: str):
        """Clean up progress data for a file_id."""
        if file_id in self.progress_data:
            del self.progress_data[file_id]


# Singleton instance
progress_manager = ProgressManager()


# --- FastAPI WebSocket endpoint ---

async def websocket_progress_endpoint(websocket: WebSocket, file_id: str):
    """
    WebSocket endpoint for receiving real-time processing progress.
    
    Usage:
        const ws = new WebSocket(`ws://localhost:8000/ws/progress/${fileId}`);
        ws.onmessage = (event) => {
            const progress = JSON.parse(event.data);
            console.log(`Progress: ${progress.percent}%`);
        };
    """
    await progress_manager.connect(websocket, file_id)
    
    try:
        while True:
            # Keep connection alive, wait for client messages (like ping)
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                # Echo back any received message as acknowledgment
                await websocket.send_json({"type": "ack", "data": data})
            except asyncio.TimeoutError:
                # Send heartbeat
                await websocket.send_json({"type": "heartbeat"})
    except WebSocketDisconnect:
        progress_manager.disconnect(websocket, file_id)


def include_websocket(app):
    """Include WebSocket endpoints in FastAPI app."""
    app.websocket("/ws/progress/{file_id}")(websocket_progress_endpoint)
    app.websocket("/ws/live-stats/{camera_id}")(websocket_live_stats_endpoint)


# --- Live Stats Manager ---

class LiveStatsManager:
    """Manages WebSocket connections for live camera stats."""
    
    def __init__(self):
        self.active_connections: Dict[str, Set[WebSocket]] = {}
        self.current_stats: Dict[str, dict] = {}
    
    async def connect(self, websocket: WebSocket, camera_id: str):
        """Accept a new WebSocket connection for a camera."""
        await websocket.accept()
        
        if camera_id not in self.active_connections:
            self.active_connections[camera_id] = set()
        
        self.active_connections[camera_id].add(websocket)
        logger.info(f"Live stats WebSocket connected for camera_id: {camera_id}")
    
    def disconnect(self, websocket: WebSocket, camera_id: str):
        """Remove a WebSocket connection."""
        if camera_id in self.active_connections:
            self.active_connections[camera_id].discard(websocket)
            if not self.active_connections[camera_id]:
                del self.active_connections[camera_id]
        logger.info(f"Live stats WebSocket disconnected for camera_id: {camera_id}")
    
    async def broadcast_stats(self, camera_id: str, stats: dict):
        """Broadcast stats update to all connected clients."""
        self.current_stats[camera_id] = stats
        
        if camera_id not in self.active_connections:
            return
        
        disconnected = set()
        for websocket in self.active_connections[camera_id]:
            try:
                await websocket.send_json(stats)
            except Exception:
                disconnected.add(websocket)
        
        for ws in disconnected:
            self.active_connections[camera_id].discard(ws)


# Singleton instance for live stats
live_stats_manager = LiveStatsManager()


async def websocket_live_stats_endpoint(websocket: WebSocket, camera_id: str):
    """
    WebSocket endpoint for receiving real-time live camera stats.
    
    Usage:
        const ws = new WebSocket(`ws://localhost:8000/ws/live-stats/${cameraId}`);
        ws.onmessage = (event) => {
            const stats = JSON.parse(event.data);
            console.log(`Occupancy: ${stats.occupancy_rate}%`);
        };
    """
    await live_stats_manager.connect(websocket, camera_id)
    
    try:
        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                await websocket.send_json({"type": "ack", "data": data})
            except asyncio.TimeoutError:
                # Send current stats as heartbeat if available
                if camera_id in live_stats_manager.current_stats:
                    await websocket.send_json(live_stats_manager.current_stats[camera_id])
                else:
                    await websocket.send_json({"type": "heartbeat"})
    except WebSocketDisconnect:
        live_stats_manager.disconnect(websocket, camera_id)


# --- Live Video Stream via WebSocket ---

import cv2
import base64
import threading
from queue import Queue, Empty

class LiveVideoStreamer:
    """Manages live video streaming via WebSocket with proper camera cleanup."""
    
    def __init__(self):
        self.active_streams: Dict[str, dict] = {}
    
    def start_stream(self, camera_id: str, source):
        """Start a video stream for a camera."""
        import os
        import time as time_module
        
        if camera_id in self.active_streams:
            self.stop_stream(camera_id)
        
        # Create output directory for recordings
        output_dir = os.path.join(os.path.dirname(__file__), "outputs", "recordings")
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate unique filename for recording
        timestamp = int(time_module.time())
        video_filename = f"live_recording_{camera_id}_{timestamp}.mp4"
        video_path = os.path.join(output_dir, video_filename)
        
        self.active_streams[camera_id] = {
            "source": source,
            "running": True,
            "frame_queue": Queue(maxsize=5),
            "stats_queue": Queue(maxsize=10),
            "thread": None,
            "cap": None,
            "video_writer": None,
            "video_path": video_path,
            "frame_count": 0
        }
        
        # Start processing thread
        thread = threading.Thread(
            target=self._process_video,
            args=(camera_id,),
            daemon=True
        )
        self.active_streams[camera_id]["thread"] = thread
        thread.start()
        
        logger.info(f"Started video stream for camera {camera_id}")
        return True
    
    def stop_stream(self, camera_id: str):
        """Stop a video stream and release camera. Returns video info."""
        video_info = None
        if camera_id in self.active_streams:
            # Save video info before stopping
            video_path = self.active_streams[camera_id].get("video_path")
            frame_count = self.active_streams[camera_id].get("frame_count", 0)
            
            self.active_streams[camera_id]["running"] = False
            # Wait for thread to finish
            thread = self.active_streams[camera_id].get("thread")
            if thread and thread.is_alive():
                thread.join(timeout=2.0)
            
            # Store last video path for retrieval
            if video_path:
                self._last_video_path = video_path
                self._last_frame_count = frame_count
                video_info = {
                    "video_path": video_path,
                    "frame_count": frame_count
                }
            
            del self.active_streams[camera_id]
            logger.info(f"Stopped video stream for camera {camera_id}")
            return video_info
        return None
    
    def get_frame(self, camera_id: str):
        """Get the latest frame for a camera."""
        if camera_id not in self.active_streams:
            return None
        try:
            return self.active_streams[camera_id]["frame_queue"].get_nowait()
        except Empty:
            return None
    
    def get_stats(self, camera_id: str):
        """Get the latest stats for a camera."""
        if camera_id not in self.active_streams:
            return None
        try:
            return self.active_streams[camera_id]["stats_queue"].get_nowait()
        except Empty:
            return None
    
    def _process_video(self, camera_id: str):
        """Process video frames in a separate thread."""
        from services.processor import ProcessingSettings, create_components
        from myutils.tracker import convert_to_deepsort_format
        import torch
        
        stream_data = self.active_streams.get(camera_id)
        if not stream_data:
            return
        
        source = stream_data["source"]
        video_path = stream_data.get("video_path")
        video_writer = None
        
        try:
            # Open camera
            cap = cv2.VideoCapture(source)
            stream_data["cap"] = cap
            
            if not cap.isOpened():
                logger.error(f"Failed to open camera {source}")
                return
            
            logger.info(f"[INFO] Camera opened for stream {camera_id}")
            print(f"[INFO] Camera opened: {source}")
            
            # Initialize video writer for recording
            fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(video_path, fourcc, fps // 2, (width, height))  # fps//2 since we skip frames
            stream_data["video_writer"] = video_writer
            print(f"[INFO] Recording to: {video_path}")
            
            # Initialize detection components
            settings = ProcessingSettings()
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print(f"[INFO] Using device: {device}")
            
            import sys
            from pathlib import Path
            sys.path.insert(0, str(Path(__file__).parent))
            from process_video import ChairOccupancyTracker
            from myutils.detector import YOLODetector
            from deep_sort_realtime.deepsort_tracker import DeepSort
            
            detector = YOLODetector(model_path=settings.model_path, device=device)
            tracker = DeepSort(max_age=50, n_init=2, nms_max_overlap=1.0, max_cosine_distance=0.4)
            occupancy_tracker = ChairOccupancyTracker(
                proximity_threshold=settings.proximity_threshold,
                occupancy_frames_threshold=settings.occupancy_frames_threshold,
                motion_blur_threshold=settings.motion_blur_threshold
            )
            
            frame_count = 0
            while stream_data.get("running", False):
                ret, frame = cap.read()
                if not ret:
                    continue
                
                # Skip frames for performance (process every 2nd frame)
                frame_count += 1
                if frame_count % 2 != 0:
                    continue
                
                # Detection
                detections = detector.detect(frame)
                person_detections = detections[detections['class'] == 0]
                chair_detections = detections[detections['class'] == 56]
                
                # Tracking
                deepsort_detections = convert_to_deepsort_format(person_detections)
                tracks = tracker.update_tracks(deepsort_detections, frame=frame)
                
                # Occupancy analysis
                chairs = occupancy_tracker.assign_chair_ids(chair_detections, frame)
                occupancy_tracker.update_occupancy(chairs, tracks)
                
                total_chairs = len(chairs)
                occupied_chairs = sum(1 for cid in chairs if occupancy_tracker.get_occupancy_status(cid))
                occupancy_rate = occupied_chairs / total_chairs if total_chairs > 0 else 0
                
                # Draw annotations
                for chair_id, chair_data in chairs.items():
                    bbox = chair_data['bbox']
                    x1, y1, x2, y2 = map(int, bbox)
                    is_occupied = occupancy_tracker.get_occupancy_status(chair_id)
                    color = (0, 255, 0) if is_occupied else (0, 0, 255)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    label = f"Chair {chair_id}"
                    if is_occupied:
                        label += " (Occupied)"
                    cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Draw people
                for track in tracks:
                    if not track.is_confirmed():
                        continue
                    bbox = track.to_ltrb()
                    x1, y1, x2, y2 = map(int, bbox)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(frame, f"Person {track.track_id}", (x1, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                
                # Write frame to video recording
                if video_writer is not None:
                    video_writer.write(frame)
                    stream_data["frame_count"] = stream_data.get("frame_count", 0) + 1
                
                # Encode frame to JPEG
                _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                frame_base64 = base64.b64encode(jpeg.tobytes()).decode('utf-8')
                
                # Put frame in queue (non-blocking)
                try:
                    # Clear old frames if queue full
                    while stream_data["frame_queue"].full():
                        stream_data["frame_queue"].get_nowait()
                    stream_data["frame_queue"].put_nowait({
                        "frame": frame_base64,
                        "stats": {
                            "total_chairs": total_chairs,
                            "occupied_chairs": occupied_chairs,
                            "occupancy_rate": occupancy_rate
                        }
                    })
                except:
                    pass
                    
        except Exception as e:
            logger.error(f"Video processing error: {e}")
            print(f"[ERROR] Video processing error: {e}")
        finally:
            # Release video writer
            if video_writer is not None:
                video_writer.release()
                print(f"[INFO] Video saved: {video_path}")
                print(f"[INFO] Recorded {stream_data.get('frame_count', 0)} frames")
            
            if stream_data.get("cap"):
                stream_data["cap"].release()
                print(f"[INFO] Camera released for stream {camera_id}")
            logger.info(f"Video processing ended for camera {camera_id}")
    
    def is_running(self, camera_id: str):
        """Check if a stream is running."""
        return camera_id in self.active_streams and self.active_streams[camera_id].get("running", False)
    
    def get_recorded_video_path(self, camera_id: str):
        """Get the path to the recorded video file."""
        if camera_id in self.active_streams:
            return self.active_streams[camera_id].get("video_path")
        return None
    
    def get_last_video_path(self):
        """Get the most recently recorded video path."""
        return getattr(self, '_last_video_path', None)


# Singleton for video streaming
live_video_streamer = LiveVideoStreamer()


async def websocket_video_stream_endpoint(websocket: WebSocket, camera_id: str):
    """
    WebSocket endpoint for streaming video frames as base64.
    
    This provides more reliable video streaming compared to MJPEG.
    Query params: ?source=0 (camera index) or ?source=rtsp://...
    """
    await websocket.accept()
    
    # Extract source from query params
    source = websocket.query_params.get("source", "0")
    logger.info(f"Video stream WebSocket connected for camera {camera_id}, source: {source}")
    
    # Parse source
    video_source = int(source) if source.isdigit() else source
    
    # Start the video stream
    live_video_streamer.start_stream(camera_id, video_source)
    
    try:
        while True:
            # Check for stop command from client
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=0.05)
                msg = json.loads(data)
                if msg.get("command") == "stop":
                    break
            except asyncio.TimeoutError:
                pass
            except:
                pass
            
            # Get and send frame
            frame_data = live_video_streamer.get_frame(camera_id)
            if frame_data:
                await websocket.send_json({
                    "type": "frame",
                    "frame": frame_data["frame"],
                    "stats": frame_data["stats"]
                })
            else:
                # Small delay if no frame available
                await asyncio.sleep(0.033)  # ~30fps max
                
            # Check if stream is still running
            if not live_video_streamer.is_running(camera_id):
                await websocket.send_json({"type": "error", "message": "Stream stopped"})
                break
                
    except WebSocketDisconnect:
        logger.info(f"Video stream WebSocket disconnected for camera {camera_id}")
    except Exception as e:
        logger.error(f"Video stream error: {e}")
    finally:
        video_info = live_video_streamer.stop_stream(camera_id)
        if video_info and video_info.get("video_path"):
            import os
            video_filename = os.path.basename(video_info["video_path"])
            try:
                await websocket.send_json({
                    "type": "complete",
                    "video_url": f"/api/recordings/{video_filename}",
                    "frame_count": video_info.get("frame_count", 0)
                })
            except:
                pass  # WebSocket might already be closed
        logger.info(f"Video stream ended for camera {camera_id}")


def include_websocket(app):
    """Include WebSocket endpoints in FastAPI app."""
    app.websocket("/ws/progress/{file_id}")(websocket_progress_endpoint)
    app.websocket("/ws/live-stats/{camera_id}")(websocket_live_stats_endpoint)
    app.websocket("/ws/video-stream/{camera_id}")(websocket_video_stream_endpoint)
