# The corrected code
import torch

class YOLODetector:
    def __init__(self, model_path, device='cpu'): # Add the 'device' parameter
        self.device = device
        # Load the model
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
        # Move the model to the selected device (GPU or CPU)
        self.model.to(self.device)
        print(f"YOLOv5 model loaded on device: {self.device}")

    def detect(self, frame):
        # Your detect logic should already be similar to this
        self.model.eval()
        with torch.no_grad():
            results = self.model(frame)
        return results.pandas().xyxy[0]