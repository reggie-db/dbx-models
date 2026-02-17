# DBX Models

MLflow PyFunc models for object detection on Databricks. These models accept base64-encoded images and return detection results with bounding boxes, labels, and confidence scores.

## Models

| Model | Description | Backend | Weights |
|-------|-------------|---------|---------|
| `yolo_object_detect` | General object detection | Ultralytics YOLO | Local `.pt` file |
| `truck_detect` | Truck detection with class filtering | SAHI + Ultralytics | Local `.pt` file |
| `hot_dog_detect` | Hot dog detection | Roboflow inference | Roboflow cloud |
| `pizza_detect` | Pizza detection | Roboflow inference | Roboflow cloud |
| `spill_detect` | Spill detection | Roboflow inference | Roboflow cloud |

## Project Structure

```
dbx-models/
├── common.ipynb              # Shared utilities (widgets, image handling, MLflow init)
├── mlfow_models.py           # Base model classes (BasePythonModel, YoloPythonModel)
├── yolo_model.py             # General YOLO detection model
├── yolo_object_detect.ipynb  # Notebook to train/register YOLO model
├── truck_detect_model.py     # Truck detection with SAHI
├── truck_detect.ipynb        # Notebook to train/register truck model
├── hot_dog_detect_model.py   # Hot dog detection via Roboflow
├── hot_dog_detect.ipynb      # Notebook to register hot dog model
├── pizza_detect_model.py     # Pizza detection via Roboflow
├── pizza_detect.ipynb        # Notebook to register pizza model
├── spill_detect_model.py     # Spill detection via Roboflow
├── spill_detect.ipynb        # Notebook to register spill model
└── test_images/              # Sample images for testing
```

## Input/Output Format

### Input

All models accept a pandas DataFrame with a single column:

| Column | Type | Description |
|--------|------|-------------|
| `image_base64` | string | Base64-encoded image (PNG, JPEG, etc.) |

### Output

All models return a pandas DataFrame with a single column:

| Column | Type | Description |
|--------|------|-------------|
| `detections` | list[dict] | List of detection dictionaries |

Each detection dictionary contains:

```python
{
    "label": str,        # Class name (e.g., "truck", "Hot Dog")
    "class_id": int,     # Numeric class ID
    "confidence": float, # Confidence score (0.0 - 1.0)
    "x1": float,         # Left edge of bounding box
    "y1": float,         # Top edge of bounding box
    "x2": float,         # Right edge of bounding box
    "y2": float          # Bottom edge of bounding box
}
```

## Configuration

### Databricks Widgets

The notebooks use Databricks widgets for configuration. Set these before running:

| Widget | Default | Description |
|--------|---------|-------------|
| `CATALOG_NAME` | `reggie_pierce` | Unity Catalog name |
| `SCHEMA_NAME` | `iot_ingest` | Schema for model registration |
| `ALIAS` | `champion` | Model alias for deployment |
| `TEST_IMAGES_PATH` | `test_images` | Path to test images |

### Environment Variables

For Roboflow-based models (hot_dog, pizza, spill):

| Variable | Description |
|----------|-------------|
| `ROBOFLOW_API_KEY` | Roboflow API key for model inference |

The API key can also be stored in Databricks secrets under the scope matching `CATALOG_NAME`.

## Usage

### Running a Notebook

1. Open the desired notebook in Databricks (e.g., `yolo_object_detect.ipynb`)
2. Configure widgets as needed
3. Run all cells to:
   - Install dependencies
   - Log the model to MLflow
   - Register the model in Unity Catalog
   - Test inference with sample images

### Loading a Registered Model

```python
import mlflow
import pandas as pd
import base64

# Load the model
model_uri = "models:/reggie_pierce.iot_ingest.yolo_object_detect@champion"
model = mlflow.pyfunc.load_model(model_uri)

# Prepare input
with open("image.jpg", "rb") as f:
    image_base64 = base64.b64encode(f.read()).decode("utf-8")

df = pd.DataFrame({"image_base64": [image_base64]})

# Run inference
result = model.predict(df)
detections = result.iloc[0]["detections"]

for det in detections:
    print(f"{det['label']}: {det['confidence']:.2f} at ({det['x1']}, {det['y1']}) - ({det['x2']}, {det['y2']})")
```

## Model Details

### YOLO Object Detect

General-purpose object detection using Ultralytics YOLO. Detects all 80 COCO classes.

- **Weights**: `yolo11n.pt` (or specify custom weights)
- **Confidence threshold**: 0.25

### Truck Detect

Specialized truck detection using SAHI (Sliced Aided Hyper Inference) for improved detection in large images.

- **Weights**: `rtdetr-l.pt` (RT-DETR model)
- **Class filter**: Only returns "truck" detections
- **Confidence threshold**: 0.20
- **Post-processing**: Removes fully-contained duplicate boxes

### Hot Dog Detect

Hot dog detection using Roboflow inference SDK.

- **Roboflow project**: `hot-dog-zxusc` (version 3)
- **Class filter**: Only returns "Hot Dog" detections
- **Label normalization**: "Salchicha Abajo" maps to "Hot Dog"
- **Confidence threshold**: 0.40

### Pizza Detect

Pizza detection using Roboflow inference SDK.

- **Roboflow project**: `pizza-r2sci` (version 23)
- **Confidence threshold**: 0.40

### Spill Detect

Spill detection using Roboflow inference SDK.

- **Roboflow project**: `spills-ax5xv` (version 2)
- **Confidence threshold**: 0.40

## Dependencies

### Local weights models (yolo, truck)

```
ultralytics
sahi  # truck_detect only
```

### Roboflow models (hot_dog, pizza, spill)

```
inference-gpu
ultralytics
```

## Development

### Adding a New Model

1. Create a new model file (e.g., `my_detect_model.py`):
   - Inherit from `YoloPythonModel`
   - Implement `load_context()` and `predict()`

2. Create a corresponding notebook (e.g., `my_detect.ipynb`):
   - Install dependencies
   - Run `%run ./common`
   - Initialize MLflow with `mlflow_init("my_detect")`
   - Log and register the model

3. Add test images to `test_images/` directory

### Base Classes

- `BasePythonModel`: Provides `run()` class method for MLflow logging
- `YoloPythonModel`: Sets up YOLO environment variables and imports
