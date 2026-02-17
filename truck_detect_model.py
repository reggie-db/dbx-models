"""
Truck detection model using Ultralytics with SAHI integration.

This model uses an Ultralytics backbone (loaded via SAHI's AutoDetectionModel) to detect
trucks in images. Detections are filtered to only include the "truck" class and
fully-contained duplicate boxes are removed.
"""

import base64
from typing import Any

import cv2
import mlflow
import numpy as np
import pandas as pd

# sys.path.append(str(Path(__file__).resolve().parents[1]))
from mlfow_models import YoloPythonModel


def _remove_fully_contained_boxes(detections: list[dict]) -> list[dict]:
    """Remove detections that are fully contained within another detection.

    Sorts detections by area (largest first) and removes any detection whose
    bounding box is completely inside a larger detection's box.

    Args:
        detections: List of detection dictionaries with x1, y1, x2, y2 keys.

    Returns:
        Filtered list with nested detections removed.
    """
    if not detections:
        return detections

    # compute area once
    def area(d):
        return max(0.0, d["x2"] - d["x1"]) * max(0.0, d["y2"] - d["y1"])

    # sort largest first
    detections = sorted(detections, key=area, reverse=True)

    kept: list[dict] = []

    for det in detections:
        x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]

        contained = False
        for k in kept:
            if (
                    x1 >= k["x1"]
                    and y1 >= k["y1"]
                    and x2 <= k["x2"]
                    and y2 <= k["y2"]
            ):
                contained = True
                break

        if not contained:
            kept.append(det)

    return kept


class TruckDetectModel(YoloPythonModel):
    """MLflow PyFunc model for truck detection using SAHI sliced prediction.

    This model uses SAHI (Sliced Aided Hyper Inference) to perform sliced prediction
    on images, which improves detection of small objects in large images. Detections
    can optionally be filtered by class (default: "truck").
    """

    # Sliced prediction parameters
    SLICE_HEIGHT = 640
    SLICE_WIDTH = 640
    OVERLAP_HEIGHT_RATIO = 0.2
    OVERLAP_WIDTH_RATIO = 0.2
    CONFIDENCE_THRESHOLD = 0.20
    CLASS_FILTER = "truck"  # Set to None to return all classes

    def load_context(self, context):
        """Load the detection model using SAHI's AutoDetectionModel.

        Args:
            context: MLflow model context containing artifact paths.
        """
        super().load_context(context)

        from sahi import AutoDetectionModel

        self.detection_model = AutoDetectionModel.from_pretrained(
            model_type="ultralytics",
            model_path=context.artifacts["weights"],
            confidence_threshold=self.CONFIDENCE_THRESHOLD,
            device="cuda:0" if self._cuda_available() else "cpu",
        )

    def _cuda_available(self) -> bool:
        """Check if CUDA is available for GPU inference."""
        try:
            import torch

            return torch.cuda.is_available()
        except ImportError:
            return False

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
        """Run truck detection on batch of images.

        Args:
            context: MLflow prediction context.
            model_input: DataFrame with 'image_base64' column.

        Returns:
            DataFrame with 'detections' column containing truck detection results.
        """
        import torch

        # -------------------------
        # 1) Decode whole batch
        # -------------------------
        decoded_images: list[np.ndarray] = []
        valid_rows: list[int] = []

        for idx, b64 in enumerate(model_input["image_base64"]):
            img = self._decode(b64)
            if img is None:
                continue
            decoded_images.append(img)
            valid_rows.append(idx)

        if not decoded_images:
            return pd.DataFrame({"detections": [[] for _ in range(len(model_input))]})

        # -------------------------
        # 2) Prepare RGB batch
        # -------------------------
        rgb_images = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in decoded_images]

        # -------------------------
        # 3) True batched inference
        # -------------------------
        # this is the key line
        results = self.detection_model.model(rgb_images, verbose=False)

        # -------------------------
        # 4) Convert detections
        # -------------------------
        batch_detections: list[list[dict[str, Any]]] = [[] for _ in range(len(model_input))]

        for batch_idx, result in enumerate(results):

            if result.boxes is None:
                continue

            boxes = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()

            detections = []

            for box, score, cls in zip(boxes, scores, classes):
                label = self.detection_model.model.names[int(cls)]

                if self.CLASS_FILTER and label.lower() != self.CLASS_FILTER:
                    continue

                detections.append({
                    "label": label,
                    "class_id": int(cls),
                    "confidence": float(score),
                    "x1": float(box[0]),
                    "y1": float(box[1]),
                    "x2": float(box[2]),
                    "y2": float(box[3]),
                })

            filtered = _remove_fully_contained_boxes(detections)

            original_row = valid_rows[batch_idx]
            batch_detections[original_row] = filtered

        return pd.DataFrame({"detections": batch_detections})


mlflow.models.set_model(TruckDetectModel())
