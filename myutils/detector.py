from ultralytics import YOLO
import pandas as pd

class YOLODetector:
    def __init__(self, model_path, device='cpu'):
        self.device = device
        # Load the YOLOv11 model using the new Ultralytics library
        self.model = YOLO(model_path) 
        print(f"YOLOv11 model loaded on device: {self.device}")

    def detect(self, frame):
        # Run inference (verbose=False keeps your terminal clean)
        results = self.model(frame, device=self.device, verbose=False)[0]
        
        # Convert the new YOLOv11 results into the Pandas format your 
        # existing app expects (xmin, ymin, xmax, ymax, confidence, class, name)
        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf)
            cls = int(box.cls)
            name = results.names[cls]
            
            detections.append({
                'xmin': x1, 'ymin': y1, 'xmax': x2, 'ymax': y2,
                'confidence': conf, 'class': cls, 'name': name
            })
            
        # Return a DataFrame. If empty, ensure columns exist to prevent crashes.
        if not detections:
            return pd.DataFrame(columns=['xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'class', 'name'])
            
        return pd.DataFrame(detections)