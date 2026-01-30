"""
API endpoint tests for Chair Occupancy Analytics.
Uses FastAPI TestClient for HTTP-level testing.
"""
import pytest
from fastapi.testclient import TestClient
from pathlib import Path
import json
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app import app


@pytest.fixture
def client():
    """Create test client for the FastAPI app."""
    with TestClient(app) as c:
        yield c


@pytest.fixture
def test_video_path():
    """Path to the test video file."""
    return Path(__file__).parent.parent / "data" / "raw_videos" / "video.mp4"


@pytest.fixture
def test_video2_path():
    """Path to the second test video file."""
    return Path(__file__).parent.parent / "data" / "raw_videos" / "video2.mp4"


class TestRootEndpoint:
    """Tests for the root endpoint."""
    
    def test_root_returns_html(self, client):
        """Root endpoint should return the HTML frontend."""
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")


class TestHealthEndpoints:
    """Tests for health and status endpoints."""
    
    def test_history_endpoint_returns_list(self, client):
        """History endpoint should return a list (even if empty)."""
        response = client.get("/api/history")
        assert response.status_code == 200
        assert isinstance(response.json(), list)


class TestVideoProcessing:
    """Tests for video processing endpoint."""
    
    def test_process_video_rejects_invalid_extension(self, client):
        """Should reject files with invalid extensions."""
        # Create a fake text file
        response = client.post(
            "/process-video",
            files={"file": ("test.txt", b"fake content", "text/plain")}
        )
        assert response.status_code == 400
        assert "Invalid file format" in response.json()["detail"]
    
    def test_process_video_rejects_missing_file(self, client):
        """Should return 422 when no file is provided."""
        response = client.post("/process-video")
        assert response.status_code == 422  # Validation error
    
    def test_process_video_accepts_valid_file(self, client, test_video_path):
        """Should accept and process a valid video file."""
        if not test_video_path.exists():
            pytest.skip(f"Test video not found: {test_video_path}")
        
        with open(test_video_path, "rb") as f:
            response = client.post(
                "/process-video",
                files={"file": ("test_video.mp4", f, "video/mp4")},
                data={
                    "proximity_threshold": 80,
                    "occupancy_frames_threshold": 5,
                    "motion_blur_threshold": 100
                }
            )
        
        # Processing can take time, but should succeed
        assert response.status_code == 200
        result = response.json()
        assert result["success"] is True
        assert "file_id" in result
        assert "processing_results" in result
    
    def test_process_video_validates_parameters(self, client, test_video_path):
        """Should clamp invalid parameter values to valid ranges."""
        if not test_video_path.exists():
            pytest.skip(f"Test video not found: {test_video_path}")
        
        with open(test_video_path, "rb") as f:
            # Pass extreme values that should be clamped
            response = client.post(
                "/process-video",
                files={"file": ("test_video.mp4", f, "video/mp4")},
                data={
                    "proximity_threshold": -100,  # Should be clamped to min
                    "occupancy_frames_threshold": 1000,  # Should be clamped to max
                    "motion_blur_threshold": 50
                }
            )
        
        assert response.status_code == 200


class TestResultsEndpoint:
    """Tests for results retrieval endpoints."""
    
    def test_get_results_not_found(self, client):
        """Should return 404 for non-existent file_id."""
        response = client.get("/api/results/nonexistent-id")
        assert response.status_code == 404
    
    def test_download_not_found(self, client):
        """Should return 404 for non-existent video download."""
        response = client.get("/download/nonexistent-id")
        assert response.status_code == 404


class TestDeleteEndpoint:
    """Tests for delete functionality."""
    
    def test_delete_not_found(self, client):
        """Should return 404 when deleting non-existent record."""
        response = client.delete("/api/history/nonexistent-id")
        assert response.status_code == 404


class TestStaticFiles:
    """Tests for static file serving."""
    
    def test_outputs_directory_accessible(self, client):
        """Outputs directory should be mounted and accessible."""
        # This will return 404 if the file doesn't exist, but shouldn't error
        response = client.get("/outputs/nonexistent.mp4")
        # 404 is expected for non-existent file, but not 500
        assert response.status_code in [404, 200]
