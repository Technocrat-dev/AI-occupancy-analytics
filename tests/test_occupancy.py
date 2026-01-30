"""
Unit tests for the ChairOccupancyTracker and related ML logic.
"""
import pytest
import numpy as np
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
from collections import defaultdict

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Try to import ML components, skip if dependencies not available
try:
    from process_video import (
        ChairOccupancyTracker,
        MultiCameraManager,
        analyze_activity_windows,
        create_interaction_ledger,
        analyze_person_metrics,
        analyze_chair_metrics
    )
    ML_AVAILABLE = True
except ImportError as e:
    ML_AVAILABLE = False
    ML_IMPORT_ERROR = str(e)

# Skip decorator for ML tests
requires_ml = pytest.mark.skipif(not ML_AVAILABLE, reason=f"ML dependencies not available")


@requires_ml
class TestChairOccupancyTracker:
    """Tests for the ChairOccupancyTracker class."""
    
    @pytest.fixture
    def tracker(self):
        """Create a fresh tracker instance for each test."""
        return ChairOccupancyTracker(
            proximity_threshold=80,
            occupancy_frames_threshold=5,
            motion_blur_threshold=100
        )
    
    def test_initialization(self, tracker):
        """Tracker should initialize with correct default values."""
        assert tracker.proximity_threshold == 80
        assert tracker.occupancy_frames_threshold == 5
        assert tracker.motion_blur_threshold == 100
        assert tracker.chair_counter == 0
    
    def test_calculate_distance(self, tracker):
        """Distance calculation should be correct."""
        # Test zero distance
        assert tracker.calculate_distance((0, 0), (0, 0)) == 0
        
        # Test 3-4-5 triangle
        assert tracker.calculate_distance((0, 0), (3, 4)) == 5.0
        
        # Test symmetric
        d1 = tracker.calculate_distance((1, 2), (5, 8))
        d2 = tracker.calculate_distance((5, 8), (1, 2))
        assert d1 == d2
    
    def test_get_center_4_element_bbox(self, tracker):
        """Should correctly calculate center for [x1, y1, x2, y2] format."""
        bbox = [0, 0, 100, 100]
        center = tracker.get_center(bbox)
        assert center == (50.0, 50.0)
        
        bbox2 = [10, 20, 30, 80]
        center2 = tracker.get_center(bbox2)
        assert center2 == (20.0, 50.0)
    
    def test_get_chair_zones(self, tracker):
        """Should properly divide chair into seat and back zones."""
        bbox = [0, 0, 100, 100]
        zones = tracker.get_chair_zones(bbox)
        
        assert 'seat' in zones
        assert 'back' in zones
        
        # Seat should be lower portion (higher y values)
        seat = zones['seat']
        back = zones['back']
        
        assert seat[1] > back[1]  # Seat top should be below back top
    
    def test_is_valid_chair_configuration(self, tracker):
        """Should validate chair configurations based on size and aspect ratio."""
        # Valid chair
        valid_bbox = [0, 0, 50, 100]  # 50x100 = 5000 area, 0.5 aspect ratio
        assert tracker.is_valid_chair_configuration(valid_bbox) is True
        
        # Too small
        tiny_bbox = [0, 0, 5, 5]  # 25 area
        assert tracker.is_valid_chair_configuration(tiny_bbox) is False
        
        # Too large
        huge_bbox = [0, 0, 1000, 1000]  # 1,000,000 area
        assert tracker.is_valid_chair_configuration(huge_bbox) is False
    
    def test_is_at_frame_edge(self, tracker):
        """Should detect when chair bbox is at frame edges."""
        frame_shape = (720, 1280, 3)  # height, width, channels
        
        # Not at edge
        center_bbox = [100, 100, 200, 200]
        assert tracker.is_at_frame_edge(center_bbox, frame_shape) is False
        
        # At left edge
        left_bbox = [0, 100, 50, 200]
        assert tracker.is_at_frame_edge(left_bbox, frame_shape) is True
        
        # At top edge
        top_bbox = [100, 0, 200, 50]
        assert tracker.is_at_frame_edge(top_bbox, frame_shape) is True
        
        # At right edge
        right_bbox = [1230, 100, 1280, 200]
        assert tracker.is_at_frame_edge(right_bbox, frame_shape) is True
        
        # At bottom edge
        bottom_bbox = [100, 670, 200, 720]
        assert tracker.is_at_frame_edge(bottom_bbox, frame_shape) is True
    
    def test_occupancy_status_requires_history(self, tracker):
        """Occupancy status should require sufficient history."""
        # No history should return False
        assert tracker.get_occupancy_status(1) is False
        
        # Add some history
        tracker.chair_occupancy_history[1] = [True, True, True]
        # Still not enough frames (need 5)
        assert tracker.get_occupancy_status(1) is False
        
        # Add more history
        tracker.chair_occupancy_history[1] = [True, True, True, True, True]
        assert tracker.get_occupancy_status(1) is True

@requires_ml
class TestMultiCameraManager:
    """Tests for the MultiCameraManager class."""
    
    @pytest.fixture
    def manager(self):
        """Create a fresh multi-camera manager."""
        return MultiCameraManager()
    
    def test_initialization(self, manager):
        """Manager should initialize with empty state."""
        assert len(manager.camera_zones) == 0
        assert len(manager.camera_priorities) == 0
        assert manager.global_chair_counter == 0
    
    def test_set_camera_zones(self, manager):
        """Should correctly set camera zones and priorities."""
        zones = [(0, 0, 640, 480)]
        manager.set_camera_zones(1, zones, priority=2)
        
        assert 1 in manager.camera_zones
        assert manager.camera_zones[1] == zones
        assert manager.camera_priorities[1] == 2
    
    def test_global_chair_id_assignment(self, manager):
        """Should assign unique global IDs to chairs."""
        id1 = manager.get_global_chair_id(1, 1)
        id2 = manager.get_global_chair_id(1, 2)
        id3 = manager.get_global_chair_id(2, 1)
        
        # All should be unique
        assert len({id1, id2, id3}) == 3
        
        # Same local ID from same camera should return same global ID
        id1_again = manager.get_global_chair_id(1, 1)
        assert id1 == id1_again
    
    def test_calculate_distance(self, manager):
        """Should calculate Euclidean distance correctly."""
        dist = manager._calculate_distance((0, 0), (3, 4))
        assert dist == 5.0

@requires_ml
class TestAnalyticsFunctions:
    """Tests for the analytics helper functions."""
    
    def test_analyze_activity_windows_empty(self):
        """Should handle empty history gracefully."""
        result = analyze_activity_windows([], 30)
        assert result["peak_window"] == "N/A"
        assert result["lowest_window"] == "N/A"
    
    def test_analyze_activity_windows_insufficient_data(self):
        """Should handle insufficient data for window analysis."""
        # Less than one window worth of data
        short_history = [{"frame": i, "occupancy_rate": 0.5} for i in range(10)]
        result = analyze_activity_windows(short_history, 30, window_seconds=60)
        assert result["peak_window"] == "N/A"
    
    def test_create_interaction_ledger_empty(self):
        """Should return empty ledger for empty history."""
        result = create_interaction_ledger([], 30)
        assert result == []
    
    def test_create_interaction_ledger_basic(self):
        """Should create ledger entries from mapping history."""
        # Simulate a person sitting in chair 1 for frames 0-90 (3 seconds at 30fps)
        mapping_history = []
        for i in range(90):
            if i < 60:
                mapping_history.append({"frame": i, "mapping": {1: [1]}})
            else:
                mapping_history.append({"frame": i, "mapping": {}})
        
        ledger = create_interaction_ledger(mapping_history, 30)
        
        assert len(ledger) == 1
        assert ledger[0]["person_id"] == 1
        assert ledger[0]["chair_id"] == 1
        assert ledger[0]["duration_sec"] == 2.0  # 60 frames / 30 fps = 2 seconds
    
    def test_analyze_person_metrics_empty(self):
        """Should return empty dict for empty ledger."""
        result = analyze_person_metrics([])
        assert result == {}
    
    def test_analyze_person_metrics_basic(self):
        """Should correctly aggregate person statistics."""
        ledger = [
            {"person_id": 1, "chair_id": 1, "start_sec": 0, "end_sec": 10, "duration_sec": 10},
            {"person_id": 1, "chair_id": 2, "start_sec": 15, "end_sec": 20, "duration_sec": 5},
        ]
        
        result = analyze_person_metrics(ledger)
        
        assert "Person 1" in result
        assert result["Person 1"]["total_interaction_time_sec"] == 15
        assert result["Person 1"]["unique_chairs_used"] == 2
        assert result["Person 1"]["session_count"] == 2
    
    def test_analyze_chair_metrics_empty(self):
        """Should return empty dict for empty ledger."""
        result = analyze_chair_metrics([])
        assert result == {}
    
    def test_analyze_chair_metrics_basic(self):
        """Should correctly aggregate chair statistics."""
        ledger = [
            {"person_id": 1, "chair_id": 1, "start_sec": 0, "end_sec": 10, "duration_sec": 10},
            {"person_id": 2, "chair_id": 1, "start_sec": 15, "end_sec": 20, "duration_sec": 5},
        ]
        
        result = analyze_chair_metrics(ledger)
        
        assert "Chair 1" in result
        assert result["Chair 1"]["total_usage_time_sec"] == 15
        assert result["Chair 1"]["unique_users"] == 2


class TestConfigValidation:
    """Tests for configuration and validation."""
    
    def test_config_validation_ranges(self):
        """Config validation should clamp values to valid ranges."""
        from config import settings
        
        # Test proximity threshold clamping
        assert settings.validate_proximity_threshold(-10) == 10  # Min
        assert settings.validate_proximity_threshold(1000) == 500  # Max
        assert settings.validate_proximity_threshold(100) == 100  # In range
        
        # Test occupancy frames clamping
        assert settings.validate_occupancy_frames(0) == 1  # Min
        assert settings.validate_occupancy_frames(100) == 60  # Max
        assert settings.validate_occupancy_frames(10) == 10  # In range
        
        # Test motion blur threshold clamping
        assert settings.validate_motion_blur_threshold(5) == 10  # Min
        assert settings.validate_motion_blur_threshold(1000) == 500  # Max
        assert settings.validate_motion_blur_threshold(150) == 150  # In range
