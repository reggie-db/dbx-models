"""
Base MLflow model classes and utilities for object detection models.

This module provides base classes for creating MLflow PyFunc models that use
YOLO-based object detection. It includes utilities for temporary directory
management and a standardized model logging workflow.
"""

import os
import pathlib
import sys
import tempfile
import uuid
from os import PathLike

import mlflow

_CODE_PATHS = [__file__]


def tmp_dir(name: PathLike | str) -> pathlib.Path:
    """Create a writable temporary directory, preferring Spark local dirs if available.

    Args:
        name: Name for the temporary directory.

    Returns:
        Path to the created temporary directory.

    Raises:
        PermissionError: If no writable directory can be found.
    """
    spark_local_dirs = os.environ.get("SPARK_LOCAL_DIRS", None)
    spark_local_dir = spark_local_dirs.split(",")[0] if spark_local_dirs else None
    for unique in [False, True]:
        append = f"_{uuid.uuid4()}" if unique else ""
        if spark_local_dir:
            if dir := _writable_dir(spark_local_dir, name, append):
                return dir
        if dir := _writable_dir(tempfile.gettempdir(), name, append):
            return dir
    raise PermissionError("No writable temporary directory found.")


def _writable_dir(*paths: PathLike | str) -> pathlib.Path | None:
    """Test if a directory path is writable by creating and removing a temp file.

    Args:
        *paths: Path components to join.

    Returns:
        The directory path if writable, None otherwise.
    """
    if paths:
        # noinspection PyBroadException
        try:
            joined_path = "/".join(pathlib.Path(p).as_posix().strip("/") for p in paths if p)
            test_file = pathlib.Path(joined_path) / f".{uuid.uuid4()}"
            test_file.parent.mkdir(parents=True, exist_ok=True)
            test_file.write_text("")
            test_file.unlink()
            return test_file.parent
        except Exception:
            pass
    return None


class BasePythonModel(mlflow.pyfunc.PythonModel):
    """Base class for MLflow PyFunc models with standardized logging.

    Provides a class method `run()` that logs the model to MLflow with
    the model file as the python_model artifact.
    """

    @classmethod
    def run(cls, **options) -> str:
        """Log the model to MLflow and return the model URI.

        Args:
            **options: Additional options passed to mlflow.pyfunc.log_model().
                Common options include artifacts, pip_requirements,
                input_example, and signature.

        Returns:
            MLflow model URI in the format "runs:/{run_id}/model".
        """
        file_path = pathlib.Path(sys.modules[cls.__module__].__file__).resolve()
        run_options: dict = {
            "artifact_path": "model",
            "python_model": file_path,
            "code_paths": _CODE_PATHS,
        }
        run_options.update(options)

        with mlflow.start_run() as mlfow_run:
            mlflow.pyfunc.log_model(**run_options)
            return f"runs:/{mlfow_run.info.run_id}/model"


class YoloPythonModel(BasePythonModel):
    """Base class for YOLO-based detection models.

    Sets up the YOLO environment variables and imports required for
    Ultralytics YOLO inference.
    """

    def load_context(self, context: mlflow.pyfunc.PythonModelContext):
        """Initialize YOLO environment and import dependencies.

        Args:
            context: MLflow model context.
        """
        super().load_context(context)
        os.environ["YOLO_CONFIG_DIR"] = str(tmp_dir("yolo_config"))
        os.environ["MODEL_CACHE_DIR"] = str("model_cache")
        from ultralytics import YOLO
