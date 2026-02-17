"""
General-purpose YOLO object detection model.

This model uses Ultralytics YOLO for inference with local weight files.
It detects all object classes supported by the loaded weights.
"""

import base64
from typing import Any

import cv2
import mlflow.pyfunc
import numpy as np
import pandas as pd

from mlfow_models import YoloPythonModel


class YoloModel(YoloPythonModel):
    """MLflow PyFunc model for general object detection using YOLO.

    Loads Ultralytics YOLO weights and runs inference on base64-encoded images.
    Returns all detected objects with bounding boxes, labels, and confidence scores.
    """

    def load_context(self, context):
        """Load YOLO model from weights artifact.

        Args:
            context: MLflow model context containing artifact paths.
        """
        super().load_context(context)
        from ultralytics import YOLO

        self.model = YOLO(context.artifacts["weights"])
        self.class_names = self.model.names  # id -> label

    def _decode(self, b64: str) -> np.ndarray:
        """Decode base64 string to BGR image array.

        Args:
            b64: Base64 encoded image string.

        Returns:
            BGR image as numpy array.
        """
        arr = np.frombuffer(base64.b64decode(b64), dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)

    def predict(self, context, model_input: pd.DataFrame) -> pd.DataFrame:
        """Run object detection on batch of images.

        Args:
            context: MLflow prediction context.
            model_input: DataFrame with 'image_base64' column.

        Returns:
            DataFrame with 'detections' column containing detection results.
        """

        images = [self._decode(b) for b in model_input["image_base64"]]
        results = self.model(images, conf=0.25, batch=len(images))

        rows: list[dict[str, Any]] = []
        for r in results:
            boxes = []
            if r.boxes is not None:
                xyxy = r.boxes.xyxy.cpu().numpy()
                conf = r.boxes.conf.cpu().numpy()
                cls = r.boxes.cls.cpu().numpy()

                for i in range(len(xyxy)):
                    class_id = int(cls[i])
                    boxes.append(
                        {
                            "label": self.class_names[class_id],
                            "class_id": class_id,
                            "confidence": float(conf[i]),
                            "x1": float(xyxy[i][0]),
                            "y1": float(xyxy[i][1]),
                            "x2": float(xyxy[i][2]),
                            "y2": float(xyxy[i][3]),
                        }
                    )

            rows.append({"detections": boxes})

        return pd.DataFrame(rows)


mlflow.models.set_model(YoloModel())
