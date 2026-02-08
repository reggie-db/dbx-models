"""
Hot dog detection model using Ultralytics YOLO with weights downloaded from Roboflow.

This model uses YOLO for inference with weights that were downloaded from Roboflow
during model packaging. No Roboflow API calls are made at inference time.
Includes label normalization for the "Salchicha Abajo" class name.
"""

import base64
import os

import cv2
import mlflow
import numpy as np
import pandas as pd

from mlfow_models import YoloPythonModel


def _normalize_class_name(raw_class: str) -> str:
    """Normalize class names, handling known label variations.

    Args:
        raw_class: Raw class name from model prediction.

    Returns:
        Normalized class name (e.g., "Salchicha Abajo" -> "Hot Dog").
    """
    normalized = raw_class.strip().lower()
    if normalized == "salchicha abajo":
        return "Hot Dog"
    return raw_class


class HotDogDetectModel(YoloPythonModel):
    """MLflow PyFunc model for hot dog detection using YOLO.

    This model uses Ultralytics YOLO for inference with weights downloaded
    from Roboflow during model packaging. No Roboflow API calls are made
    at inference time.
    """

    ROBOFLOW_PROJECT = "hot-dog-zxusc"
    ROBOFLOW_VERSION = 3
    MODEL_ID = f"{ROBOFLOW_PROJECT}/{ROBOFLOW_VERSION}"
    CONFIDENCE_THRESHOLD = 0.40

    def load_context(self, context):
        """Load the YOLO model from downloaded weights.

        Supports both PyTorch (.pt) and ONNX (.onnx) weight formats.
        Ultralytics YOLO can load both formats directly.

        Args:
            context: MLflow model context containing artifact paths.
        """
        super().load_context(context)

        from inference import get_model
        self.model = get_model(
            model_id=HotDogDetectModel.MODEL_ID,
            api_key=os.environ["ROBOFLOW_API_KEY"],
        )

    def _decode(self, b64: str) -> np.ndarray:
        """Decode base64 string to BGR image array.

        Args:
            b64: Base64 encoded image string.

        Returns:
            BGR image as numpy array.
        """
        if not b64:
            return None
        arr = np.frombuffer(base64.b64decode(b64), dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)

    def predict(self, context, model_input: pd.DataFrame) -> pd.DataFrame:

        images = []
        row_map = []

        for idx, b64 in enumerate(model_input["image_base64"]):
            img = self._decode(b64)
            if img is None:
                continue
            images.append(img)
            row_map.append(idx)

        if not images:
            return pd.DataFrame({"detections": [[] for _ in range(len(model_input))]})

        results = self.model.infer(images, confidence=self.CONFIDENCE_THRESHOLD)

        output = [[] for _ in range(len(model_input))]

        for result_idx, image_result in enumerate(results):
            detections = []

            for p in image_result.predictions:
                label = _normalize_class_name(p.class_name)
                if "hot dog" != label.lower():
                    continue
                cx, cy, w, h = p.x, p.y, p.width, p.height

                detections.append({
                    "label": label,
                    "class_id": int(p.class_id),
                    "confidence": float(p.confidence),
                    "x1": float(cx - w / 2),
                    "y1": float(cy - h / 2),
                    "x2": float(cx + w / 2),
                    "y2": float(cy + h / 2),
                })

            output[row_map[result_idx]] = detections

        return pd.DataFrame({"detections": output})


mlflow.models.set_model(HotDogDetectModel())
