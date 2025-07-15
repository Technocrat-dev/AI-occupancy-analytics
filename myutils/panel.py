from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2
import pandas as pd
import numpy as np
from collections import defaultdict
def draw_chair_person_mapping_panel(frame, chairs, occupancy_tracker):
    """Draw a compact panel showing chair-person mapping and unoccupied chairs"""
    # Panel dimensions
    panel_width = 100
    line_height = 18
    padding = 5
    
    # Count occupied and unoccupied chairs
    occupied_chairs = len(occupancy_tracker.chair_person_mapping)
    unoccupied_chairs = []
    
    # Find unoccupied chairs
    for chair_id in chairs.keys():
        if not occupancy_tracker.get_occupancy_status(chair_id):
            unoccupied_chairs.append(chair_id)
    
    # Calculate panel height based on content
    total_items = occupied_chairs + len(unoccupied_chairs) + 2  # +2 for headers
    panel_height = total_items * line_height + padding * 2
    
    # Create panel background
    panel = np.ones((panel_height, panel_width, 3), dtype=np.uint8) * 40
    
    y_offset = 16
    
    # Show occupied chairs
    if occupied_chairs > 0:
        cv2.putText(panel, "Occupied:", (padding, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        y_offset += line_height
        
        for chair_id, person_id in occupancy_tracker.chair_person_mapping.items():
            text = f"C{chair_id}->P{person_id}"
            cv2.putText(panel, text, (padding, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
            y_offset += line_height
    
    # Show unoccupied chairs
    if unoccupied_chairs:
        cv2.putText(panel, "Vacant:", (padding, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        y_offset += line_height
        
        # Show unoccupied chair IDs
        vacant_text = ",".join([f"C{chair_id}" for chair_id in sorted(unoccupied_chairs)])
        cv2.putText(panel, vacant_text, (padding, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
    
    # If no chairs at all
    if occupied_chairs == 0 and len(unoccupied_chairs) == 0:
        cv2.putText(panel, "No chairs", (padding, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (128, 128, 128), 1)
    
    # Overlay panel on frame
    frame[10:10+panel_height, 10:10+panel_width] = panel
    
    return frame