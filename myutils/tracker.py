def convert_to_deepsort_format(detections):
    """Convert YOLO11 detections to DeepSORT format"""
    deepsort_detections = []
    
    for _, detection in detections.iterrows():
        x1, y1, x2, y2 = (
            float(detection['xmin']),
            float(detection['ymin']),
            float(detection['xmax']),
            float(detection['ymax']),
        )
        width = x2 - x1
        height = y2 - y1
        confidence = float(detection['confidence']) if 'confidence' in detection else 0.5
        detection_class = int(detection['class'])
        
        if width > 0 and height > 0:
            deepsort_detections.append(([x1, y1, width, height], confidence, detection_class))
    
    return deepsort_detections