import mlflow.pyfunc
from typing import Any
import pandas as pd
import numpy as np
import base64
import json
import cv2


class YoloModel(mlflow.pyfunc.PythonModel):

    def load_context(self, context):
        import os
        import tempfile
        from pathlib import Path

        yolo_config_dir = (
            Path(tempfile.gettempdir())
        / "yolo_config"
        )
        yolo_config_dir.mkdir(parents=True, exist_ok=True)
        os.environ["YOLO_CONFIG_DIR"] = str(yolo_config_dir)

        from ultralytics import YOLO

        self.model = YOLO(context.artifacts["weights"])
        self.class_names = self.model.names  # id -> label

    def _decode(self, b64: str):
        arr = np.frombuffer(base64.b64decode(b64), dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)

    def predict(self, context, model_input: pd.DataFrame) -> pd.DataFrame:

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