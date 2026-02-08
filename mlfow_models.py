import os
import pathlib
import sys
import tempfile
import uuid
from os import PathLike

import mlflow

_CODE_PATHS = [__file__]


def tmp_dir(name: PathLike | str):
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

    @classmethod
    def run(cls, **options):
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

    def load_context(self, context: mlflow.pyfunc.PythonModelContext):
        super().load_context(context)
        os.environ["YOLO_CONFIG_DIR"] = str(tmp_dir("yolo_config"))
        os.environ["MODEL_CACHE_DIR"] = str("model_cache")
        from ultralytics import YOLO
