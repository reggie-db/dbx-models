import base64
from typing import Any

import cv2
import mlflow.pyfunc
import numpy as np
import pandas as pd

#sys.path.append(str(Path(__file__).resolve().parents[1]))
from mlfow_models import YoloPythonModel


class YoloModel(YoloPythonModel):

    def load_context(self, context):
        super().load_context(context)
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
