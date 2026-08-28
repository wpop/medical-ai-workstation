"""Validate non-golden real studies across Python and C++ deployment paths."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort  # type: ignore[import-untyped]
from scipy.special import softmax

PATIENT_IDS = ("patient002", "patient022", "patient041", "patient061", "patient081")
CPP_RESULT_PREFIX = "MAIW_REAL_STUDY_RESULT "
EXPECTED_LOGITS_SHAPE = (1, 5)
DEFAULT_CPP_TIMEOUT_SECONDS = 30.0


def positive_finite_float(value: str) -> float:
    """Parse a finite positive floating-point command-line value."""
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be finite and greater than zero")
    return parsed


def parse_args() -> argparse.Namespace:
    """Parse explicit external repository, artifact, and executable paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python-repository", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--deployment-package", type=Path, required=True)
    parser.add_argument("--cpp-classifier", type=Path, required=True)
    parser.add_argument(
        "--cpp-timeout-seconds",
        type=positive_finite_float,
        default=DEFAULT_CPP_TIMEOUT_SECONDS,
        help=f"Per-patient C++ classifier timeout (default: {DEFAULT_CPP_TIMEOUT_SECONDS:g})",
    )
    return parser.parse_args()


def load_python_contract(python_repository: Path) -> tuple[Any, Any, Any]:
    """Import the authoritative indexer and preprocessing classes without copying them."""
    source_dir = python_repository / "src"
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Python source directory is missing: {source_dir}")
    sys.path.insert(0, str(source_dir))

    from cardiac_pathology.data import AcdcDatasetIndexer  # noqa: PLC0415
    from cardiac_pathology.preprocessing import (  # noqa: PLC0415
        PatientPreprocessor,
        PreprocessingConfig,
    )

    return AcdcDatasetIndexer, PatientPreprocessor, PreprocessingConfig


def require_object(value: Any, name: str) -> dict[str, Any]:
    """Require a JSON object at a deployment metadata location."""
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    return value


def load_deployment_metadata(package_dir: Path) -> dict[str, Any]:
    """Load deployment metadata that defines preprocessing and ONNX contracts."""
    metadata_path = package_dir / "deployment.json"
    with metadata_path.open("r", encoding="utf-8") as file:
        return require_object(json.load(file), "deployment metadata")


def build_preprocessing_config(config_type: Any, metadata: dict[str, Any]) -> Any:
    """Construct the authoritative Python config from packaged frozen values."""
    preprocessing = require_object(metadata.get("preprocessing"), "preprocessing")
    spacing = preprocessing.get("target_spacing_xyz_mm")
    shape = preprocessing.get("final_shape_dhw")
    percentiles = preprocessing.get("clip_percentiles")
    if not isinstance(spacing, list) or len(spacing) != 3:
        raise ValueError("target_spacing_xyz_mm must contain three values")
    if not isinstance(shape, list) or len(shape) != 3:
        raise ValueError("final_shape_dhw must contain three values")
    if not isinstance(percentiles, list) or len(percentiles) != 2:
        raise ValueError("clip_percentiles must contain two values")
    return config_type(
        target_orientation=str(preprocessing["target_orientation"]),
        target_spacing_xyz=tuple(float(value) for value in spacing),
        target_shape_dhw=tuple(int(value) for value in shape),
        lower_percentile=float(percentiles[0]),
        upper_percentile=float(percentiles[1]),
        epsilon=float(preprocessing["normalization_epsilon"]),
        z_padding_value=float(preprocessing["z_padding_value"]),
    )


def run_python_inference(
    session: ort.InferenceSession,
    input_name: str,
    output_name: str,
    tensor: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Run deployment ONNX inference on an authoritative preprocessed tensor."""
    outputs = session.run([output_name], {input_name: np.expand_dims(tensor, axis=0)})
    if len(outputs) != 1 or not isinstance(outputs[0], np.ndarray):
        raise ValueError("Python ONNX Runtime returned an invalid output collection")
    logits = outputs[0]
    if logits.shape != EXPECTED_LOGITS_SHAPE or logits.dtype != np.float32:
        raise ValueError(f"Python logits contract mismatch: {logits.shape}, {logits.dtype}")
    if not np.isfinite(logits).all():
        raise ValueError("Python logits contain non-finite values")
    probabilities = softmax(logits.astype(np.float64), axis=1)[0]
    return logits[0], probabilities, int(np.argmax(logits, axis=1)[0])


def run_cpp_inference(
    executable: Path,
    ed_path: Path,
    es_path: Path,
    patient_id: str,
    timeout_seconds: float,
) -> dict[str, Any]:
    """Run the existing C++ classification service smoke executable for one study."""
    try:
        completed = subprocess.run(
            [str(executable), str(ed_path), str(es_path)],
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(
            f"{patient_id}: C++ classifier exceeded the {timeout_seconds:g}-second timeout"
        ) from error
    result_lines = [
        line.removeprefix(CPP_RESULT_PREFIX)
        for line in completed.stdout.splitlines()
        if line.startswith(CPP_RESULT_PREFIX)
    ]
    if len(result_lines) != 1:
        raise ValueError("C++ classifier did not emit exactly one structured result")
    return require_object(json.loads(result_lines[0]), "C++ result")


def validate_patient(
    *,
    patient: Any,
    preprocessor: Any,
    session: ort.InferenceSession,
    input_name: str,
    output_name: str,
    class_names: tuple[str, ...],
    cpp_classifier: Path,
    cpp_timeout_seconds: float,
    logit_tolerance: float,
) -> tuple[float, float, int, str]:
    """Compare one patient's independent Python and production C++ results."""
    tensor = preprocessor.preprocess(patient).tensor
    python_logits, python_probabilities, python_prediction = run_python_inference(
        session, input_name, output_name, tensor
    )
    cpp_result = run_cpp_inference(
        cpp_classifier,
        patient.ed_path,
        patient.es_path,
        patient.patient_id,
        cpp_timeout_seconds,
    )
    cpp_logits = np.asarray(cpp_result.get("logits"), dtype=np.float64)
    cpp_probabilities = np.asarray(cpp_result.get("probabilities"), dtype=np.float64)
    cpp_prediction = cpp_result.get("predicted_class_index")
    cpp_class_name = cpp_result.get("predicted_class_name")
    if cpp_logits.shape != (5,) or cpp_probabilities.shape != (5,):
        raise ValueError(f"{patient.patient_id}: C++ result arrays must contain five values")
    if not np.isfinite(cpp_logits).all() or not np.isfinite(cpp_probabilities).all():
        raise ValueError(f"{patient.patient_id}: C++ result contains non-finite values")

    max_logit_error = float(np.max(np.abs(cpp_logits - python_logits.astype(np.float64))))
    max_probability_error = float(np.max(np.abs(cpp_probabilities - python_probabilities)))
    probability_tolerance = (0.5 * logit_tolerance) + 1e-12
    expected_class_name = class_names[python_prediction]
    parity_passed = (
        max_logit_error <= logit_tolerance
        and max_probability_error <= probability_tolerance
        and cpp_prediction == python_prediction
        and cpp_class_name == expected_class_name
    )
    if not parity_passed:
        raise ValueError(
            f"{patient.patient_id}: parity failed; logit_error={max_logit_error:.9e}, "
            f"probability_error={max_probability_error:.9e}, "
            f"python_class={python_prediction}, cpp_class={cpp_prediction}"
        )
    return max_logit_error, max_probability_error, python_prediction, expected_class_name


def main() -> None:
    """Run the fixed five-patient non-golden engineering parity cohort."""
    args = parse_args()
    indexer_type, preprocessor_type, config_type = load_python_contract(args.python_repository)
    metadata = load_deployment_metadata(args.deployment_package)
    onnx = require_object(metadata.get("onnx"), "onnx")
    onnx_input = require_object(onnx.get("input"), "onnx input")
    onnx_output = require_object(onnx.get("output"), "onnx output")
    validation = require_object(metadata.get("validation"), "validation")
    classes = metadata.get("classes")
    if not isinstance(classes, list) or len(classes) != 5:
        raise ValueError("deployment classes must contain five entries")
    class_objects = tuple(require_object(item, "class") for item in classes)
    if any(class_object.get("index") != index for index, class_object in enumerate(class_objects)):
        raise ValueError("deployment classes must be ordered by contiguous indices 0..4")
    class_names = tuple(str(class_object["name"]) for class_object in class_objects)

    class_mapping_path = args.python_repository / "configs/class_mapping.json"
    indexer = indexer_type(args.dataset_dir, class_mapping_path)
    patients_by_id = {patient.patient_id: patient for patient in indexer.index_patients()}
    preprocessor = preprocessor_type(build_preprocessing_config(config_type, metadata))
    session = ort.InferenceSession(
        str(args.deployment_package / str(onnx["filename"])),
        providers=["CPUExecutionProvider"],
    )
    logit_tolerance = float(validation["absolute_logit_tolerance"])

    print("patient     group  predicted  max_logit_error  max_probability_error  status")
    for patient_id in PATIENT_IDS:
        patient = patients_by_id[patient_id]
        logit_error, probability_error, prediction, class_name = validate_patient(
            patient=patient,
            preprocessor=preprocessor,
            session=session,
            input_name=str(onnx_input["name"]),
            output_name=str(onnx_output["name"]),
            class_names=class_names,
            cpp_classifier=args.cpp_classifier,
            cpp_timeout_seconds=args.cpp_timeout_seconds,
            logit_tolerance=logit_tolerance,
        )
        print(
            f"{patient_id:<11} {patient.class_name:<6} {prediction}:{class_name:<8} "
            f"{logit_error:.9e}  {probability_error:.9e}  PASS"
        )
    print("\nFive-patient Python/C++ engineering parity: PASS")


if __name__ == "__main__":
    main()
