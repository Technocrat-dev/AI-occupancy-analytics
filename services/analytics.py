"""
Analytics functions for chair occupancy data.
Extracted from process_video.py for modularity.
"""
from collections import defaultdict
from typing import List, Dict, Any


def analyze_activity_windows(occupancy_history: List[Dict], fps: float, window_seconds: int = 60) -> Dict[str, Any]:
    """
    Analyzes occupancy history to find peak and off-peak windows.
    
    Args:
        occupancy_history: List of dicts with 'frame' and 'occupancy_rate' keys
        fps: Frames per second of the video
        window_seconds: Size of the analysis window in seconds
        
    Returns:
        Dict with peak_window, peak_rate, lowest_window, lowest_rate
    """
    if not occupancy_history or not fps or len(occupancy_history) < int(window_seconds * fps):
        return {"peak_window": "N/A", "peak_rate": 0, "lowest_window": "N/A", "lowest_rate": 0}

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


def create_interaction_ledger(mapping_history: List[Dict], fps: float) -> List[Dict]:
    """
    Creates a detailed log of every person-chair interaction from frame history.
    
    Args:
        mapping_history: List of dicts with 'frame' and 'mapping' keys
        fps: Frames per second of the video
        
    Returns:
        List of interaction records with person_id, chair_id, start_sec, end_sec, duration_sec
    """
    ledger = []
    active_sessions = {}  # {chair_id: {'person_id': ..., 'start_frame': ...}}

    for frame_data in mapping_history:
        frame_num = frame_data['frame']
        current_mapping = frame_data['mapping']  # {chair_id: [person_id, ...]}

        # Check for ended sessions
        for chair_id in list(active_sessions.keys()):
            if chair_id not in current_mapping or not current_mapping[chair_id]:
                session = active_sessions.pop(chair_id)
                duration_sec = (frame_num - session['start_frame']) / fps
                if duration_sec > 1:  # Only log sessions longer than 1 second
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
                    'person_id': person_ids[0],  # Track the first person
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


def analyze_person_metrics(ledger: List[Dict]) -> Dict[str, Dict]:
    """
    Analyzes the ledger to provide per-person statistics.
    
    Args:
        ledger: Interaction ledger from create_interaction_ledger()
        
    Returns:
        Dict mapping person labels to their metrics
    """
    if not ledger:
        return {}
    
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


def analyze_chair_metrics(ledger: List[Dict]) -> Dict[str, Dict]:
    """
    Analyzes the ledger to provide per-chair statistics.
    
    Args:
        ledger: Interaction ledger from create_interaction_ledger()
        
    Returns:
        Dict mapping chair labels to their metrics
    """
    if not ledger:
        return {}
    
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
