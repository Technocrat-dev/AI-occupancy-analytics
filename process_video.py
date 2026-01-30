from myutils.detector import YOLODetector
from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2
import numpy as np
from collections import defaultdict
import math
import time
import datetime 
from myutils.panel import draw_chair_person_mapping_panel
from myutils.tracker import convert_to_deepsort_format
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

class MultiCameraManager:
    def __init__(self):
        """Enhanced multi-camera manager with chair deduplication and conflict resolution"""
        self.camera_zones = {}
        self.camera_priorities = {}
        self.global_chair_registry = {}
        self.camera_to_global_id = defaultdict(dict)
        self.global_chair_counter = 0
        self.overlap_regions = []
        self.chair_duplicate_threshold = 100
        self.unified_occupancy_map = {}

    def set_camera_zones(self, camera_id, zones, priority=1):
        """Set priority zones for a camera with priority level"""
        self.camera_zones[camera_id] = zones
        self.camera_priorities[camera_id] = priority

    def add_overlap_region(self, region, camera_ids):
        """Define overlapping regions between cameras"""
        self.overlap_regions.append({
            'region': region,
            'cameras': camera_ids,
            'priority_camera': max(camera_ids, key=lambda x: self.camera_priorities.get(x, 1))
        })

    def is_chair_authoritative(self, camera_id, chair_bbox):
        """Enhanced authority check with overlap resolution"""
        chair_center = ((chair_bbox[0] + chair_bbox[2]) / 2, (chair_bbox[1] + chair_bbox[3]) / 2)

        for overlap in self.overlap_regions:
            region = overlap['region']
            if (region[0] <= chair_center[0] <= region[2] and
                region[1] <= chair_center[1] <= region[3]):
                return camera_id == overlap['priority_camera']

        if camera_id not in self.camera_zones:
            return True

        for zone in self.camera_zones[camera_id]:
            x1, y1, x2, y2 = zone
            if x1 <= chair_center[0] <= x2 and y1 <= chair_center[1] <= y2:
                return True
        return False

    def detect_duplicate_chairs(self, all_camera_chairs):
        """Detect and resolve duplicate chairs across cameras"""
        duplicates = []
        chairs_list = []

        for cam_id, chairs in all_camera_chairs.items():
            for chair_id, chair_data in chairs.items():
                chairs_list.append({
                    'camera_id': cam_id,
                    'local_id': chair_id,
                    'center': chair_data['center'],
                    'bbox': chair_data['bbox'],
                    'data': chair_data
                })

        for i, chair1 in enumerate(chairs_list):
            for j, chair2 in enumerate(chairs_list[i+1:], i+1):
                if chair1['camera_id'] != chair2['camera_id']:
                    distance = self._calculate_distance(chair1['center'], chair2['center'])
                    if distance < self.chair_duplicate_threshold:
                        # FIX #3: Hysteresis / Robust Priority Logic
                        # We stick strictly to defined priorities to prevent flickering (Z-fighting)
                        priority_cam = max(chair1['camera_id'], chair2['camera_id'],
                                         key=lambda x: self.camera_priorities.get(x, 1))
                        duplicates.append({
                            'chair1': chair1,
                            'chair2': chair2,
                            'priority_camera': priority_cam,
                            'distance': distance
                        })

        return duplicates

    def resolve_chair_conflicts(self, all_camera_chairs):
        """Resolve conflicts and create unified chair map"""
        duplicates = self.detect_duplicate_chairs(all_camera_chairs)
        resolved_chairs = {}
        excluded_chairs = set()

        for dup in duplicates:
            chair1 = dup['chair1']
            chair2 = dup['chair2']
            priority_cam = dup['priority_camera']

            if chair1['camera_id'] != priority_cam:
                excluded_chairs.add((chair1['camera_id'], chair1['local_id']))
            if chair2['camera_id'] != priority_cam:
                excluded_chairs.add((chair2['camera_id'], chair2['local_id']))

        for cam_id, chairs in all_camera_chairs.items():
            resolved_chairs[cam_id] = {}
            for chair_id, chair_data in chairs.items():
                if (cam_id, chair_id) not in excluded_chairs:
                    resolved_chairs[cam_id][chair_id] = chair_data

        return resolved_chairs

    def get_unified_occupancy_count(self, all_camera_occupancy):
        """Get unified occupancy count across all cameras"""
        total_occupied = 0
        total_chairs = 0

        resolved_chairs = self.resolve_chair_conflicts(all_camera_occupancy)

        for cam_id, chairs in resolved_chairs.items():
            for chair_id, is_occupied in chairs.items():
                total_chairs += 1
                if is_occupied:
                    total_occupied += 1

        self.unified_occupancy_map = {
            'total_chairs': total_chairs,
            'occupied_chairs': total_occupied,
            'occupancy_rate': total_occupied / total_chairs if total_chairs > 0 else 0
        }

        return self.unified_occupancy_map

    def get_global_chair_id(self, camera_id, local_chair_id):
        """Get or create global chair ID with deduplication"""
        if local_chair_id in self.camera_to_global_id[camera_id]:
            return self.camera_to_global_id[camera_id][local_chair_id]
        else:
            self.global_chair_counter += 1
            global_id = self.global_chair_counter
            self.camera_to_global_id[camera_id][local_chair_id] = global_id
            return global_id

    def _calculate_distance(self, point1, point2):
        """Calculate Euclidean distance between two points"""
        return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

class ChairOccupancyTracker:
    def __init__(self, proximity_threshold=50, occupancy_frames_threshold=5, motion_blur_threshold=100, camera_id=1, multi_cam_manager=None):
        """
        Initialize Chair Occupancy Tracker with enhanced edge case handling
        """
        self.proximity_threshold = proximity_threshold
        self.occupancy_frames_threshold = occupancy_frames_threshold
        self.motion_blur_threshold = motion_blur_threshold
        self.chair_person_mapping = defaultdict(list)
        self.chair_occupancy_history = defaultdict(list)
        self.chair_counter = 0
        self.chair_positions = {}


        self.camera_id = camera_id
        self.multi_cam_manager = multi_cam_manager
        self.is_multi_camera = multi_cam_manager is not None

        self.person_height_threshold = 1.8
        self.chair_zones = {}
        self.dwell_time_tracking = defaultdict(int)

        self.previous_brightness = None
        self.brightness_change_threshold = 30
        self.reflection_zones = []
        self.chair_last_known_positions = {}
        

        self.chair_movement_vectors = defaultdict(list)
        self.chair_color_signatures = {}
        self.chair_lifecycle_states = {}
        self.valid_chair_aspect_ratios = (0.2, 5.0)
        self.chair_size_range = (100, 100000)

    def calculate_distance(self, point1, point2):
        """Calculate Euclidean distance between two points"""
        return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

    def get_center(self, bbox):
        """Get center point of bounding box"""
        if len(bbox) == 4:
            return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
        else:
            return (bbox[0] + bbox[2]/2, bbox[1] + bbox[3]/2)

    def get_person_reference_point(self, person_bbox, chair_center):
        """Get appropriate reference point based on person's position and posture"""
        person_height = person_bbox[3] - person_bbox[1]
        person_width = person_bbox[2] - person_bbox[0]
        height_width_ratio = person_height / person_width if person_width > 0 else 0

        person_center = self.get_center(person_bbox)
        person_bottom = (person_center[0], person_bbox[3])

        if height_width_ratio > self.person_height_threshold:
            return person_bottom

        if person_center[1] < chair_center[1] - 20:
            return person_bottom

        return person_center

    def get_chair_zones(self, chair_bbox):
        """Divide chair into seat and back regions for partial occupancy"""
        x1, y1, x2, y2 = chair_bbox
        width = x2 - x1
        height = y2 - y1

        seat_zone = [x1, y1 + int(0.4 * height), x2, y2]
        back_zone = [x1, y1, x2, y1 + int(0.6 * height)]

        return {'seat': seat_zone, 'back': back_zone}

    def check_zone_occupancy(self, person_center, chair_zones):
        """Check which zones of the chair the person occupies"""
        occupied_zones = []

        for zone_name, zone_bbox in chair_zones.items():
            zone_center = self.get_center(zone_bbox)
            distance = self.calculate_distance(person_center, zone_center)

            if distance < self.proximity_threshold * 0.7:
                occupied_zones.append(zone_name)

        return occupied_zones

    def detect_brightness_change(self, frame):
        """Detect extreme lighting changes frame-to-frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        current_brightness = np.mean(gray)

        if self.previous_brightness is not None:
            brightness_diff = abs(current_brightness - self.previous_brightness)
            if brightness_diff > self.brightness_change_threshold:
                return True, brightness_diff

        self.previous_brightness = current_brightness
        return False, 0

    def enhance_frame_adaptive(self, frame, brightness_change):
        """Apply adaptive enhancement based on lighting conditions"""
        if brightness_change > self.brightness_change_threshold:
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
            lab[:,:,0] = clahe.apply(lab[:,:,0])
            frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        return frame

    def detect_reflections(self, frame):
        """Detect and filter reflections using symmetry analysis"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        _, bright_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(bright_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        reflection_zones = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 100:
                x, y, w, h = cv2.boundingRect(contour)
                reflection_zones.append([x, y, x+w, y+h])

        self.reflection_zones = reflection_zones
        return reflection_zones

    def is_in_reflection_zone(self, bbox):
        """Check if detection is in a reflection zone"""
        detection_center = self.get_center(bbox)

        for ref_zone in self.reflection_zones:
            if (ref_zone[0] <= detection_center[0] <= ref_zone[2] and
                ref_zone[1] <= detection_center[1] <= ref_zone[3]):
                return True
        return False

    def detect_motion_blur(self, frame):
        """Detect motion blur using Laplacian variance"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        return laplacian_var < self.motion_blur_threshold

    def interpolate_chair_position(self, chair_id):
        """Interpolate chair position when occluded"""
        if chair_id in self.chair_movement_vectors and len(self.chair_movement_vectors[chair_id]) > 0:
            last_movement = self.chair_movement_vectors[chair_id][-1]
            last_position = self.chair_last_known_positions.get(chair_id)

            if last_position:
                predicted_pos = (
                    last_position[0] + last_movement[0],
                    last_position[1] + last_movement[1]
                )
                return predicted_pos

        return self.chair_last_known_positions.get(chair_id)

    def extract_chair_signature(self, frame, bbox):
        """Extract enhanced color and spatial signature for chair tracking"""
        x1, y1, x2, y2 = map(int, bbox)
        roi = frame[y1:y2, x1:x2]

        if roi.size == 0:
            return None

        hist_b = cv2.calcHist([roi], [0], None, [32], [0, 256])
        hist_g = cv2.calcHist([roi], [1], None, [32], [0, 256])
        hist_r = cv2.calcHist([roi], [2], None, [32], [0, 256])

        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray_roi, 50, 150)
        edge_density = np.sum(edges > 0) / (roi.shape[0] * roi.shape[1])

        return {
            'color_hist': np.concatenate([hist_b.flatten(), hist_g.flatten(), hist_r.flatten()]),
            'edge_density': edge_density,
            'aspect_ratio': (x2-x1) / (y2-y1) if (y2-y1) > 0 else 1.0
        }

    def compare_chair_signatures(self, sig1, sig2):
        """Compare chair signatures for matching"""
        if sig1 is None or sig2 is None:
            return float('inf')

        color_corr = cv2.compareHist(sig1['color_hist'].reshape(-1, 1),
                                   sig2['color_hist'].reshape(-1, 1),
                                   cv2.HISTCMP_CORREL)

        edge_diff = abs(sig1['edge_density'] - sig2['edge_density'])

        aspect_diff = abs(sig1['aspect_ratio'] - sig2['aspect_ratio'])

        similarity = (1 - color_corr) + edge_diff + aspect_diff
        return similarity

    def is_valid_chair_configuration(self, bbox):
        """Check if chair configuration is valid (not stacked/folded)"""
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        area = width * height
        aspect_ratio = width / height if height > 0 else 0

        if not (self.chair_size_range[0] <= area <= self.chair_size_range[1]):
            return False

        if not (self.valid_chair_aspect_ratios[0] <= aspect_ratio <= self.valid_chair_aspect_ratios[1]):
            return False

        return True

    def is_at_frame_edge(self, bbox, frame_shape):
        """Check if chair is partially visible at frame edges"""
        h, w = frame_shape[:2]
        edge_threshold = 10

        return (bbox[0] <= edge_threshold or bbox[1] <= edge_threshold or
                bbox[2] >= w - edge_threshold or bbox[3] >= h - edge_threshold)

    def assign_chair_ids(self, chair_detections, frame):
        """Enhanced chair ID assignment with movement tracking and lifecycle management"""
        current_chairs = {}
        frame_height, frame_width = frame.shape[:2]

        for _, chair in chair_detections.iterrows():
            chair_bbox = [chair['xmin'], chair['ymin'], chair['xmax'], chair['ymax']]

            if self.is_multi_camera and not self.multi_cam_manager.is_chair_authoritative(self.camera_id, chair_bbox):
                continue

            chair_center = self.get_center(chair_bbox)
            chair_signature = self.extract_chair_signature(frame, chair_bbox)

            matched_id = None
            min_distance = float('inf')
            best_signature_match = float('inf')

            for chair_id, stored_data in self.chair_positions.items():
                if self.chair_lifecycle_states.get(chair_id) == 'removed':
                    continue

                stored_center = stored_data if isinstance(stored_data, tuple) else stored_data.get('center')
                distance = self.calculate_distance(chair_center, stored_center)

                position_match = distance < 50

                signature_match = False
                if chair_id in self.chair_color_signatures and chair_signature:
                    sig_similarity = self.compare_chair_signatures(
                        self.chair_color_signatures[chair_id], chair_signature)
                    signature_match = sig_similarity < 1.0

                if (position_match or signature_match) and distance < min_distance:
                    min_distance = distance
                    matched_id = chair_id
                    best_signature_match = sig_similarity if signature_match else distance

            if matched_id is not None:
                old_center = self.chair_positions[matched_id]
                if isinstance(old_center, dict):
                    old_center = old_center['center']

                # FIX #2: Weighted Average to prevent drift
                alpha = 0.1 # Smoothing factor (0.1 means we only move 10% towards the new detection)
                smooth_x = alpha * chair_center[0] + (1 - alpha) * old_center[0]
                smooth_y = alpha * chair_center[1] + (1 - alpha) * old_center[1]
                smoothed_center = (smooth_x, smooth_y)
                
                movement_vector = (smoothed_center[0] - old_center[0],
                                 smoothed_center[1] - old_center[1])
                self.chair_movement_vectors[matched_id].append(movement_vector)

                if len(self.chair_movement_vectors[matched_id]) > 5:
                    self.chair_movement_vectors[matched_id].pop(0)

                current_chairs[matched_id] = {
                    'bbox': chair_bbox,
                    'center': smoothed_center,
                    'zones': self.get_chair_zones(chair_bbox),
                    'at_edge': self.is_at_frame_edge(chair_bbox, (frame_height, frame_width))
                }

                self.chair_positions[matched_id] = smoothed_center
                self.chair_last_known_positions[matched_id] = smoothed_center
                self.chair_lifecycle_states[matched_id] = 'active'

                if chair_signature:
                    self.chair_color_signatures[matched_id] = chair_signature

            else:
                self.chair_counter += 1
                chair_id = self.chair_counter

                current_chairs[chair_id] = {
                    'bbox': chair_bbox,
                    'center': chair_center,
                    'zones': self.get_chair_zones(chair_bbox),
                    'at_edge': self.is_at_frame_edge(chair_bbox, (frame_height, frame_width))
                }

                self.chair_positions[chair_id] = chair_center
                self.chair_last_known_positions[chair_id] = chair_center
                self.chair_lifecycle_states[chair_id] = 'active'

                if chair_signature:
                    self.chair_color_signatures[chair_id] = chair_signature

        for chair_id in list(self.chair_positions.keys()):
            if chair_id not in current_chairs and self.chair_lifecycle_states.get(chair_id) == 'active':
                interpolated_pos = self.interpolate_chair_position(chair_id)
                if interpolated_pos:
                    current_chairs[chair_id] = {
                        'bbox': [interpolated_pos[0]-25, interpolated_pos[1]-25,
                                interpolated_pos[0]+25, interpolated_pos[1]+25],
                        'center': interpolated_pos,
                        'zones': {},
                        'at_edge': False,
                        'interpolated': True
                    }
                    self.chair_lifecycle_states[chair_id] = 'occluded'
                else:
                    self.chair_lifecycle_states[chair_id] = 'inactive'

        return current_chairs

    def update_occupancy(self, chairs, tracks):
        """Enhanced occupancy update with complex behavior handling"""
        frame_occupancy = {}
        current_mapping = defaultdict(list)

        for chair_id, chair_data in chairs.items():
            chair_center = chair_data['center']
            chair_zones = chair_data.get('zones', {})
            nearby_people = []

            if chair_zones:
                self.chair_zones[chair_id] = chair_zones

            for track in tracks:
                if not track.is_confirmed():
                    continue

                person_bbox = track.to_ltrb()
                person_ref_point = self.get_person_reference_point(person_bbox, chair_center)
                distance = self.calculate_distance(chair_center, person_ref_point)

                if distance < self.proximity_threshold:
                    occupied_zones = []
                    if chair_zones:
                        occupied_zones = self.check_zone_occupancy(person_ref_point, chair_zones)

                    nearby_people.append({
                        'id': track.track_id,
                        'distance': distance,
                        'zones': occupied_zones,
                        'bbox': person_bbox
                    })

                    self.dwell_time_tracking[f"{chair_id}_{track.track_id}"] += 1

            nearby_people.sort(key=lambda x: x['distance'])

            assigned_people = nearby_people[:2]
            current_mapping[chair_id] = [p['id'] for p in assigned_people]

            is_occupied = len(assigned_people) > 0
            frame_occupancy[chair_id] = is_occupied

            self.chair_occupancy_history[chair_id].append(is_occupied)

            if len(self.chair_occupancy_history[chair_id]) > 30:
                self.chair_occupancy_history[chair_id].pop(0)

        for chair_id in current_mapping:
            recent_occupancy = self.chair_occupancy_history[chair_id][-self.occupancy_frames_threshold:]
            
            # FIX #5: Occupancy Logic Tolerance
            if len(recent_occupancy) >= self.occupancy_frames_threshold:
                # Calculate percentage of occupied frames
                occupancy_score = sum(recent_occupancy) / len(recent_occupancy)
                # If 80% of frames were occupied, we count it (allows for flickering)
                if occupancy_score > 0.8:
                    self.chair_person_mapping[chair_id] = current_mapping[chair_id]

        chairs_to_remove = []
        for chair_id in self.chair_person_mapping:
            if chair_id in self.chair_occupancy_history:
                recent_occupancy = self.chair_occupancy_history[chair_id][-self.occupancy_frames_threshold:]
                # Same tolerance logic for removal
                if len(recent_occupancy) >= self.occupancy_frames_threshold:
                    occupancy_score = sum(recent_occupancy) / len(recent_occupancy)
                    if occupancy_score < 0.2: # If less than 20% occupied, clear it
                        chairs_to_remove.append(chair_id)

        for chair_id in chairs_to_remove:
            del self.chair_person_mapping[chair_id]

        return frame_occupancy

    def get_occupancy_status(self, chair_id):
        """Get stable occupancy status for a chair"""
        if chair_id not in self.chair_occupancy_history:
            return False

        recent_history = self.chair_occupancy_history[chair_id][-5:]
        if len(recent_history) < 3:
            return False

        return sum(recent_history) > len(recent_history) / 2

# FIX #4: Removed duplicate analyze_activity_windows function here

## MODIFICATION: New helper functions for advanced analytics

def analyze_activity_windows(occupancy_history, fps, window_seconds=60):
    """Analyzes occupancy history to find peak and off-peak windows."""
    if not occupancy_history or not fps or len(occupancy_history) < int(window_seconds * fps):
        return { "peak_window": "N/A", "peak_rate": 0, "lowest_window": "N/A", "lowest_rate": 0 }

    window_frames = int(window_seconds * fps)
    max_rate, min_rate = -1.0, 2.0
    peak_start_frame, lowest_start_frame = 0, 0

    for i in range(len(occupancy_history) - window_frames):
        window = occupancy_history[i : i + window_frames]
        avg_rate = sum(h['occupancy_rate'] for h in window) / len(window)
        if avg_rate > max_rate:
            max_rate = avg_rate
            peak_start_frame = occupancy_history[i]['frame']
        if avg_rate < min_rate:
            min_rate = avg_rate
            lowest_start_frame = occupancy_history[i]['frame']

    def format_time(frame):
        total_seconds = int(frame / fps)
        minutes, seconds = divmod(total_seconds, 60)
        return f"{minutes}m {seconds}s"

    return {
        "peak_window": f"{format_time(peak_start_frame)} - {format_time(peak_start_frame + window_frames)}",
        "peak_rate": max_rate,
        "lowest_window": f"{format_time(lowest_start_frame)} - {format_time(lowest_start_frame + window_frames)}",
        "lowest_rate": min_rate,
    }

def create_interaction_ledger(mapping_history, fps):
    """Creates a detailed log of every person-chair interaction from frame history."""
    ledger = []
    active_sessions = {}  # {chair_id: {'person_id': ..., 'start_frame': ...}}

    for frame_data in mapping_history:
        frame_num = frame_data['frame']
        current_mapping = frame_data['mapping'] # {chair_id: [person_id, ...]}

        # Check for ended sessions
        for chair_id in list(active_sessions.keys()):
            if chair_id not in current_mapping or not current_mapping[chair_id]:
                session = active_sessions.pop(chair_id)
                duration_sec = (frame_num - session['start_frame']) / fps
                if duration_sec > 1: # Only log sessions longer than 1 second
                    ledger.append({
                        "person_id": session['person_id'],
                        "chair_id": chair_id,
                        "start_sec": round(session['start_frame'] / fps, 1),
                        "end_sec": round(frame_num / fps, 1),
                        "duration_sec": round(duration_sec, 1)
                    })

        # Check for new sessions
        for chair_id, person_ids in current_mapping.items():
            if person_ids and chair_id not in active_sessions:
                active_sessions[chair_id] = {
                    'person_id': person_ids[0], # Track the first person
                    'start_frame': frame_num
                }

    # End any remaining sessions at the end of the video
    if mapping_history:
        for chair_id, session in active_sessions.items():
            duration_sec = (mapping_history[-1]['frame'] - session['start_frame']) / fps
            if duration_sec > 1:
                ledger.append({
                    "person_id": session['person_id'],
                    "chair_id": chair_id,
                    "start_sec": round(session['start_frame'] / fps, 1),
                    "end_sec": round(mapping_history[-1]['frame'] / fps, 1),
                    "duration_sec": round(duration_sec, 1)
                })
    return ledger

def analyze_person_metrics(ledger):
    """Analyzes the ledger to provide per-person statistics."""
    if not ledger: return {}
    person_stats = defaultdict(lambda: {"total_time": 0, "chairs_used": set(), "moves": 0})
    for session in ledger:
        pid = session['person_id']
        person_stats[pid]['total_time'] += session['duration_sec']
        person_stats[pid]['chairs_used'].add(session['chair_id'])
        person_stats[pid]['moves'] += 1
    
    # Final formatting
    final_metrics = {}
    for pid, stats in person_stats.items():
        final_metrics[f"Person {pid}"] = {
            "total_interaction_time_sec": round(stats['total_time'], 1),
            "unique_chairs_used": len(stats['chairs_used']),
            "session_count": stats['moves']
        }
    return final_metrics

def analyze_chair_metrics(ledger):
    """Analyzes the ledger to provide per-chair statistics."""
    if not ledger: return {}
    chair_stats = defaultdict(lambda: {"total_time": 0, "users": set()})
    for session in ledger:
        cid = session['chair_id']
        chair_stats[cid]['total_time'] += session['duration_sec']
        chair_stats[cid]['users'].add(session['person_id'])

    # Final formatting
    final_metrics = {}
    for cid, stats in chair_stats.items():
        final_metrics[f"Chair {cid}"] = {
            "total_usage_time_sec": round(stats['total_time'], 1),
            "unique_users": len(stats['users'])
        }
    return final_metrics

def process_video_for_api(video_path, output_video_path, settings=None):
    """
    Process video for API - returns data instead of showing windows
    """
    # Default settings
    if settings is None:
        settings = {
            'proximity_threshold': 80,
            'occupancy_frames_threshold': 5,
            'motion_blur_threshold': 100,
            'use_multi_camera': False
        }
    
    # Initialize components
    import torch

    # This line automatically selects the NVIDIA GPU if it's available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"--- Using device: {device} ---") 

    # FIX #4b: Point to the new YOLOv11 model file
    detector = YOLODetector(model_path="yolo11m.pt", device=device)
    tracker = DeepSort(max_age=50, n_init=2, nms_max_overlap=1.0, max_cosine_distance=0.4)
    occupancy_tracker = ChairOccupancyTracker(
        proximity_threshold=settings.get('proximity_threshold', 80), 
        occupancy_frames_threshold=settings.get('occupancy_frames_threshold', 5),
        motion_blur_threshold=settings.get('motion_blur_threshold', 100)
    )
    
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties for output video
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    frame_count = 0
    processing_stats = {
        'total_frames': total_frames,
        'processed_frames': 0,
        'total_chairs': 0,
        'max_occupied_chairs': 0,
        'occupancy_history': [],
        'processing_time': 0
    }
    ## MODIFICATION: Add new list to store mapping history for ledger
    person_chair_map_history = []
    
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        brightness_changed, brightness_diff = occupancy_tracker.detect_brightness_change(frame)
        if brightness_changed:
            frame = occupancy_tracker.enhance_frame_adaptive(frame, brightness_diff)
        
        if occupancy_tracker.detect_motion_blur(frame):
            continue
        
        occupancy_tracker.detect_reflections(frame)
        
        detections = detector.detect(frame)
        person_detections = detections[detections['class'] == 0]
        chair_detections = detections[detections['class'] == 56]
        
        deepsort_detections = convert_to_deepsort_format(person_detections)
        tracks = tracker.update_tracks(deepsort_detections, frame=frame)
        
        chairs = occupancy_tracker.assign_chair_ids(chair_detections, frame)
        frame_occupancy = occupancy_tracker.update_occupancy(chairs, tracks)
        
        # Draw on frame
        for chair_id, chair_data in chairs.items():
            bbox = chair_data['bbox']
            x1, y1, x2, y2 = map(int, bbox)
            is_occupied = occupancy_tracker.get_occupancy_status(chair_id)
            chair_color = (0, 0, 255) # Default Empty
            if is_occupied: chair_color = (0, 255, 0) # Occupied
            if chair_data.get('interpolated', False): chair_color = (255, 165, 0)
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), chair_color, 2)
            label = f"Chair {chair_id}"
            if is_occupied:
                person_id = occupancy_tracker.chair_person_mapping.get(chair_id, ['?'])[0]
                label += f" (P:{person_id})"
            cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, chair_color, 2)
        
        for track in tracks:
            if not track.is_confirmed(): continue
            bbox, track_id = track.to_ltrb(), track.track_id
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, f"Person {track_id}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        total_chairs = len(chairs)
        occupied_chairs = sum(1 for cid in chairs if occupancy_tracker.get_occupancy_status(cid))
        occupancy_rate = occupied_chairs / total_chairs if total_chairs > 0 else 0
        
        # Add stats text to frame
        stats_text = [f"Chairs: {total_chairs}", f"Occupied: {occupied_chairs}", f"Rate: {occupancy_rate:.1%}"]
        for i, text in enumerate(stats_text):
            cv2.putText(frame, text, (10, 30 + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        out.write(frame)
            
        # Collect Stats
        processing_stats['processed_frames'] = frame_count
        processing_stats['total_chairs'] = max(processing_stats['total_chairs'], total_chairs)
        processing_stats['max_occupied_chairs'] = max(processing_stats['max_occupied_chairs'], occupied_chairs)
        processing_stats['occupancy_history'].append({'frame': frame_count, 'occupancy_rate': occupancy_rate})
        
        ## MODIFICATION: Store the detailed mapping for this frame
        person_chair_map_history.append({'frame': frame_count, 'mapping': occupancy_tracker.chair_person_mapping.copy()})
        
        frame_count += 1
    
    # Cleanup
    cap.release()
    out.release()
    
    # Calculate final stats
    end_time = time.time()
    processing_stats['processing_time'] = end_time - start_time
    
    avg_occupancy = sum(h['occupancy_rate'] for h in processing_stats['occupancy_history']) / len(processing_stats['occupancy_history']) if processing_stats['occupancy_history'] else 0
    
    ## MODIFICATION: Generate all new analytics from the collected data
    interaction_ledger = create_interaction_ledger(person_chair_map_history, fps)

    # FIX #1: Clear memory leak immediately after use
    del person_chair_map_history

    person_analytics = analyze_person_metrics(interaction_ledger)
    chair_analytics = analyze_chair_metrics(interaction_ledger)
    activity_windows = analyze_activity_windows(processing_stats['occupancy_history'], fps)
    
    # Return final results
    final_stats = {
        'total_chairs_detected': processing_stats['total_chairs'],
        'max_occupied_chairs': processing_stats['max_occupied_chairs'],
        'average_occupancy_rate': avg_occupancy,
        'total_frames_processed': processing_stats['processed_frames'],
        'processing_time_seconds': processing_stats['processing_time'],
        'output_video_path': output_video_path,
        'occupancy_over_time': processing_stats['occupancy_history'],
        # Add all new analytics to the final payload
        'interaction_ledger': interaction_ledger,
        'person_analytics': person_analytics,
        'chair_analytics': chair_analytics,
        'activity_windows': activity_windows
    }
    
    return final_stats

def main():
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'api':
        return

    USE_MULTI_CAMERA = False

    if USE_MULTI_CAMERA:

        multi_cam_manager = MultiCameraManager()
        multi_cam_manager.set_camera_zones(1, [(0, 0, 640, 720)], priority=2)
        multi_cam_manager.set_camera_zones(2, [(640, 0, 1280, 720)], priority=1)
        multi_cam_manager.add_overlap_region([600, 0, 680, 720], [1, 2])

        cameras = {
            1: {
                'path': "data/raw_videos/video1.mp4",
                # FIX #4a: Updated Path
                'detector': YOLODetector(model_path="yolo11m.pt"),
                'tracker': DeepSort(max_age=50, n_init=2, nms_max_overlap=1.0, max_cosine_distance=0.4),
                'occupancy_tracker': ChairOccupancyTracker(proximity_threshold=80, occupancy_frames_threshold=5,
                                                         camera_id=1, multi_cam_manager=multi_cam_manager)
            },
            2: {
                'path': "data/raw_videos/video2.mp4",
                # FIX #4a: Updated Path
                'detector': YOLODetector(model_path="yolo11m.pt"),
                'tracker': DeepSort(max_age=50, n_init=2, nms_max_overlap=1.0, max_cosine_distance=0.4),
                'occupancy_tracker': ChairOccupancyTracker(proximity_threshold=80, occupancy_frames_threshold=5,
                                                         camera_id=2, multi_cam_manager=multi_cam_manager)
            }
        }

        caps = {cam_id: cv2.VideoCapture(cam_data['path']) for cam_id, cam_data in cameras.items()}

    else:
        # FIX #4a: Updated Path
        detector = YOLODetector(model_path="yolo11m.pt")
        tracker = DeepSort(max_age=50, n_init=2, nms_max_overlap=1.0, max_cosine_distance=0.4),
        # Note: Tracker instantiation in original was slightly bugged with comma, fixed here
        tracker = DeepSort(max_age=50, n_init=2, nms_max_overlap=1.0, max_cosine_distance=0.4)
        occupancy_tracker = ChairOccupancyTracker(proximity_threshold=80, occupancy_frames_threshold=5)

        video_path = "data/raw_videos/video.mp4"
        cap = cv2.VideoCapture(video_path)

    frame_count = 0

    while True:
        if USE_MULTI_CAMERA:

            frames = {}
            all_chairs = {}
            all_occupancy = {}

            for cam_id, cap in caps.items():
                ret, frame = cap.read()
                if not ret:
                    break
                frames[cam_id] = frame

            if not frames:
                break

            for cam_id, frame in frames.items():
                cam_data = cameras[cam_id]


                brightness_changed, brightness_diff = cam_data['occupancy_tracker'].detect_brightness_change(frame)
                if brightness_changed:
                    frame = cam_data['occupancy_tracker'].enhance_frame_adaptive(frame, brightness_diff)

                if cam_data['occupancy_tracker'].detect_motion_blur(frame):
                    continue

                detections = cam_data['detector'].detect(frame)
                person_detections = detections[detections['class'] == 0]
                chair_detections = detections[detections['class'] == 56]

                deepsort_detections = convert_to_deepsort_format(person_detections)
                tracks = cam_data['tracker'].update_tracks(deepsort_detections, frame=frame)

                chairs = cam_data['occupancy_tracker'].assign_chair_ids(chair_detections, frame)
                frame_occupancy = cam_data['occupancy_tracker'].update_occupancy(chairs, tracks)

                all_chairs[cam_id] = chairs
                all_occupancy[cam_id] = {chair_id: cam_data['occupancy_tracker'].get_occupancy_status(chair_id)
                                       for chair_id in chairs.keys()}


            resolved_chairs = multi_cam_manager.resolve_chair_conflicts(all_chairs)
            unified_stats = multi_cam_manager.get_unified_occupancy_count(all_occupancy)


            for cam_id, frame in frames.items():
                chairs = resolved_chairs.get(cam_id, {})
                cam_data = cameras[cam_id]

                for chair_id, chair_data in chairs.items():
                    bbox = chair_data['bbox']
                    x1, y1, x2, y2 = map(int, bbox)

                    is_occupied = cam_data['occupancy_tracker'].get_occupancy_status(chair_id)
                    if chair_data.get('interpolated', False):
                        chair_color = (255, 165, 0)
                    elif chair_data.get('at_edge', False):
                        chair_color = (255, 255, 0)
                    elif is_occupied:
                        chair_color = (0, 0, 255)
                    else:
                        chair_color = (0, 255, 0)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), chair_color, 2)


                    global_id = multi_cam_manager.get_global_chair_id(cam_id, chair_id)
                    label = f"Chair {global_id} {'(Occupied)' if is_occupied else '(Empty)'}"
                    cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, chair_color, 2)


                for track in cameras[cam_id]['tracker'].tracker.tracks:
                    if not track.is_confirmed():
                        continue
                    bbox = track.to_ltrb()
                    track_id = track.track_id
                    x1, y1, x2, y2 = map(int, bbox)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(frame, f"Person {track_id}", (x1, y1-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)


                stats_text = [
                    f"Cam {cam_id} - Chairs: {len(chairs)}",
                    f"Global Stats:",
                    f"Total Chairs: {unified_stats['total_chairs']}",
                    f"Occupied: {unified_stats['occupied_chairs']}",
                    f"Rate: {unified_stats['occupancy_rate']:.1%}"
                ]

                for i, text in enumerate(stats_text):
                    y_pos = 30 + i * 25
                    cv2.putText(frame, text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


                cam_data = cameras[cam_id]
                frame = draw_chair_person_mapping_panel(frame, chairs, cam_data['occupancy_tracker'])


            if frame_count % 30 == 0:
                print(f"Frame {frame_count} Multi-Camera Stats:")
                for cam_id in cameras.keys():
                    cam_chairs = len(resolved_chairs.get(cam_id, {}))
                    cam_occupied = sum(1 for occupied in all_occupancy.get(cam_id, {}).values() if occupied)
                    print(f"  Cam {cam_id}: {cam_chairs} chairs, {cam_occupied} occupied")
                print(f"  Global: {unified_stats['total_chairs']} total, {unified_stats['occupied_chairs']} occupied, {unified_stats['occupancy_rate']:.1%} rate")


                cv2.imshow(f'Camera {cam_id}', frame)


            if len(frames) == 2:
                cam1_frame = cv2.resize(frames[1], (640, 360))
                cam2_frame = cv2.resize(frames[2], (640, 360))
                combined_frame = np.vstack([cam1_frame, cam2_frame])


                cv2.putText(combined_frame, f"UNIFIED VIEW - Total: {unified_stats['total_chairs']}, "
                           f"Occupied: {unified_stats['occupied_chairs']}",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                cv2.imshow('Multi-Camera View', combined_frame)

        else:

            ret, frame = cap.read()
            if not ret:
                break


            brightness_changed, brightness_diff = occupancy_tracker.detect_brightness_change(frame)
            if brightness_changed:
                frame = occupancy_tracker.enhance_frame_adaptive(frame, brightness_diff)


            if occupancy_tracker.detect_motion_blur(frame):
                continue


            occupancy_tracker.detect_reflections(frame)


            detections = detector.detect(frame)
            person_detections = detections[detections['class'] == 0]
            chair_detections = detections[detections['class'] == 56]


            deepsort_detections = convert_to_deepsort_format(person_detections)
            tracks = tracker.update_tracks(deepsort_detections, frame=frame)


            chairs = occupancy_tracker.assign_chair_ids(chair_detections, frame)
            frame_occupancy = occupancy_tracker.update_occupancy(chairs, tracks)


            for chair_id, chair_data in chairs.items():
                bbox = chair_data['bbox']
                x1, y1, x2, y2 = map(int, bbox)

                is_occupied = occupancy_tracker.get_occupancy_status(chair_id)


                if chair_data.get('interpolated', False):
                    chair_color = (255, 165, 0)
                elif chair_data.get('at_edge', False):
                    chair_color = (255, 255, 0)
                elif is_occupied:
                    chair_color = (0, 255, 0)
                else:
                    chair_color = (0, 0, 255)

                cv2.rectangle(frame, (x1, y1), (x2, y2), chair_color, 2)

                label = f"Chair {chair_id} {'(Occupied)' if is_occupied else '(Empty)'}"
                cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, chair_color, 2)


            for track in tracks:
                if not track.is_confirmed():
                    continue
                bbox = track.to_ltrb()
                track_id = track.track_id
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, f"Person {track_id}", (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)


            total_chairs = len(chairs)
            occupied_chairs = sum(1 for chair_id in chairs.keys()
                                if occupancy_tracker.get_occupancy_status(chair_id))
            occupancy_rate = occupied_chairs / total_chairs if total_chairs > 0 else 0

            stats_text = [
                f"Total Chairs: {total_chairs}",
                f"Occupied: {occupied_chairs}",
                f"Empty: {total_chairs - occupied_chairs}",
                f"Occupancy Rate: {occupancy_rate:.1%}"
            ]

            for i, text in enumerate(stats_text):
                cv2.putText(frame, text, (10, 30 + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


            frame = draw_chair_person_mapping_panel(frame, chairs, occupancy_tracker)


            if frame_count % 30 == 0:
                print(f"Frame {frame_count}: {total_chairs} chairs, {occupied_chairs} occupied, {occupancy_rate:.1%} rate")

            cv2.imshow('Chair Occupancy Tracker', frame)

        frame_count += 1


        if cv2.waitKey(1) == 27:
            break


        if frame_count % 2 == 0:
            continue


    if USE_MULTI_CAMERA:
        for cap in caps.values():
            cap.release()
    else:
        cap.release()

    cv2.destroyAllWindows()


    if USE_MULTI_CAMERA:
        print("\n=== MULTI-CAMERA SESSION SUMMARY ===")
        print(f"Final Unified Statistics:")
        print(f"Total Chairs Detected: {unified_stats['total_chairs']}")
        print(f"Occupied Chairs: {unified_stats['occupied_chairs']}")
        print(f"Overall Occupancy Rate: {unified_stats['occupancy_rate']:.1%}")

        for cam_id, cam_data in cameras.items():
            chairs_count = len(cam_data['occupancy_tracker'].chair_positions)
            print(f"Camera {cam_id}: {chairs_count} chairs tracked")
    else:
        print("\n=== SINGLE CAMERA SESSION SUMMARY ===")
        total_chairs = len(occupancy_tracker.chair_positions)
        print(f"Total Chairs Tracked: {total_chairs}")
        print(f"Chair-Person Mappings: {len(occupancy_tracker.chair_person_mapping)}")

if __name__ == "__main__":
    main()