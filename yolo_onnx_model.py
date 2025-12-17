import mlflow.pyfunc
from typing import Any
import pandas as pd
import numpy as np
import base64
import json
import cv2


class YoloOnnxModel(mlflow.pyfunc.PythonModel):

    def load_context(self, context):
        import onnxruntime as ort

        self.session = ort.InferenceSession(
            context.artifacts["model"],
            providers=["CPUExecutionProvider"],
        )

        # classes.json has string keys
        with open(context.artifacts["classes"]) as f:
            self.class_names = json.load(f)

        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]
        self.imgsz = 480

    def _decode(self, b64: str):
        arr = np.frombuffer(base64.b64decode(b64), dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)

    def _preprocess(self, img):
        orig_h, orig_w = img.shape[:2]

        # SIMPLE resize ONLY
        img_resized = cv2.resize(img, (self.imgsz, self.imgsz))
        img_resized = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_resized = img_resized.astype("float32") / 255.0
        img_resized = img_resized.transpose(2, 0, 1)[None, ...]

        # scale factors back to original
        sx = orig_w / self.imgsz
        sy = orig_h / self.imgsz

        return img_resized, sx, sy

    def _postprocess(self, outputs, orig_w, orig_h, conf_threshold=0.25):
        preds = outputs[0]
        boxes = []

        for det in preds[0]:
            cx, cy, bw, bh, conf_raw, cls = [float(det[i]) for i in range(6)]

            conf = conf_raw / 100.0 if conf_raw > 1.0 else conf_raw
            conf = max(0.0, min(1.0, conf))

            if conf < conf_threshold:
                continue

            # normalized center -> pixel xyxy
            cx *= orig_w
            cy *= orig_h
            bw *= orig_w
            bh *= orig_h

            x1 = cx - bw / 2
            y1 = cy - bh / 2
            x2 = cx + bw / 2
            y2 = cy + bh / 2

            class_id = int(cls)
            label = self.class_names.get(str(class_id), "unknown")

            boxes.append(
                {
                    "label": label,
                    "confidence": conf,
                    "x1": float(x1),
                    "y1": float(y1),
                    "x2": float(x2),
                    "y2": float(y2),
                }
            )

        return boxes

    def predict(self, context, model_input: pd.DataFrame) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []

        for b64 in model_input["image_base64"]:
            img = self._decode(b64)
            orig_h, orig_w = img.shape[:2]

            tensor, _, _ = self._preprocess(img)

            outputs = self.session.run(
                self.output_names,
                {self.input_name: tensor},
            )

            rows.append(
                {"detections": self._postprocess(outputs, orig_w, orig_h)}
            )

        return pd.DataFrame(rows)


mlflow.models.set_model(YoloOnnxModel())