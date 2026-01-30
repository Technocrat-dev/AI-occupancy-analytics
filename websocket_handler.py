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
    """Include WebSocket endpoint in FastAPI app."""
    app.websocket("/ws/progress/{file_id}")(websocket_progress_endpoint)
