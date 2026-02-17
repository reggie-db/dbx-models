"""
Spill detection model using Roboflow inference SDK.

This model uses the Roboflow inference SDK to detect spills in images.
The model is loaded from Roboflow at runtime using the ROBOFLOW_API_KEY
environment variable.
"""

import base64
import os

import cv2
import mlflow
import numpy as np
import pandas as pd

from mlfow_models import YoloPythonModel



class SpillDetectModel(YoloPythonModel):
    """MLflow PyFunc model for spill detection using Roboflow inference.

    This model uses the Roboflow inference SDK to detect spills.
    Requires ROBOFLOW_API_KEY environment variable to be set.
    """

    ROBOFLOW_PROJECT = "spills-ax5xv"
    ROBOFLOW_VERSION = 2
    MODEL_ID = f"{ROBOFLOW_PROJECT}/{ROBOFLOW_VERSION}"
    CONFIDENCE_THRESHOLD = 0.40

    def load_context(self, context):
        """Load the model from Roboflow inference SDK.

        Args:
            context: MLflow model context.
        """
        super().load_context(context)

        from inference import get_model
        self.model = get_model(
            model_id=SpillDetectModel.MODEL_ID,
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
        """Run spill detection on batch of images.

        Args:
            context: MLflow prediction context.
            model_input: DataFrame with 'image_base64' column.

        Returns:
            DataFrame with 'detections' column containing spill detection results.
        """
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
                label = p.class_name
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


mlflow.models.set_model(SpillDetectModel())
