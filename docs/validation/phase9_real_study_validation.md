# Phase 9 Real-Study Validation

Phase 9 adds validation infrastructure and evidence for the existing cardiac MRI
classification pipeline. It does not redesign production preprocessing, inference,
or Qt workflow architecture.

## Independent Non-Golden Python/C++ Parity

The fixed five-patient cohort was evaluated with independent Python production
preprocessing and deployment ONNX Runtime, then compared with the production C++
classification service.

| Patient | ACDC Group | Production Prediction | Max Logit Error | Max Probability Error |
| --- | --- | --- | ---: | ---: |
| patient002 | DCM | DCM | 0.0 | 0.0 |
| patient022 | HCM | HCM | 0.0 | 0.0 |
| patient041 | MINF | MINF | 0.0 | 0.0 |
| patient061 | NOR | NOR | 0.0 | 0.0 |
| patient081 | RV | RV | 0.0 | 0.0 |

ACDC Group is contextual information, not the engineering parity acceptance criterion.

Engineering parity is Python production preprocessing plus deployment ONNX Runtime
versus production C++ classification. The validation harness invokes the existing
classification-service smoke executable with runtime ED/ES paths and applies a
finite, positive per-patient process timeout.

## Manual Real GUI Validation

The actual `medical-ai-workstation` Qt executable was manually exercised without
restarting between studies:

- patient002 -> DCM — PASS
- patient022 -> HCM — PASS
- patient041 -> MINF — PASS

For each study, ED and ES were selected through the existing Browse controls and
Classify completed through the asynchronous workflow. The UI remained usable, the
predicted class and all five probabilities were displayed, and the displayed class
matched the independently validated Phase 9 result. Repeated classification worked
without restarting the application, with no observed crash or hang.

## Automated Lifecycle Validation

Commit `9ebcbd7` validates the existing Qt asynchronous lifecycle with real configured
ED/ES inputs. Coverage includes:

- empty ED and empty ES validation without starting a worker;
- asynchronous medical-volume loading failure and state recovery;
- asynchronous success and failure delivery on the main Qt thread;
- concurrent request rejection while the original classification continues;
- success followed by success and failure followed by success;
- running-state transitions and event-loop responsiveness;
- recovery of the ED editor, ED Browse, ES editor, ES Browse, and Classify controls;
- destruction during active work, including waiting for the worker before workflow
  destruction returns, with an external CTest timeout.

## Regression Status

The final observed Phase 9 regression evidence is:

- active Medical AI Workstation CTests: **13/13 PASS**;
- five-patient Python/C++ parity: **5/5 PASS**;
- maximum logit error: **0.0**;
- maximum probability error: **0.0**;
- `git diff --check`: **PASS**.

When qt-viewer-pro is included through the current source-integration configuration,
its tests remain visible in the integrated CTest registry but are marked disabled by
Medical AI Workstation CMake. This is intentional integration behavior, not a test
failure.

## DICOM Status

- **NIfTI real-study workflow — VALIDATED**
- **DICOM series loading semantics — IMPLEMENTED / SOURCE-AUDITED**
- **Real cardiac ED/ES DICOM workflow — NOT VALIDATED**

No suitable real cardiac ED/ES DICOM pair was available for Phase 9. Source audit
also identified two deferred qt-viewer-pro dependency limitations:

1. The workstation picker advertises `.dicom`, while the current registry path
   recognizes `.dcm`.
2. An unmatched selected DICOM file can fall back to the first discovered series.

These are deferred dependency issues. Phase 9 does not modify qt-viewer-pro.

## CI and Local Validation Boundary

The hosted workflow has two jobs:

- **Portable Ubuntu CI** configures with `MAIW_ENABLE_QTVIEWERPRO_INTEGRATION=OFF`
  and with deployment and real-volume paths empty. It builds the portable project
  surface and runs `cardiac-mri-classification-contract`. It does not validate
  qt-viewer-pro integration, deployment artifacts, or real medical studies.
- **Qt Integration CI** checks out the pinned qt-viewer-pro revision, enables
  `MAIW_ENABLE_QTVIEWERPRO_INTEGRATION`, and builds the Qt/medical-image integration.
  Its active workstation tests are `cardiac-mri-classification-contract`,
  `cardiac-mri-preprocessing-math`, `cardiac-mri-classification-result-widget`, and
  `qt-viewer-pro-dependency-smoke`. Deployment-package, ACDC-training, and real ED/ES
  paths remain empty, so window, real-data, classification-service, lifecycle,
  shutdown, and golden parity tests that require those inputs are not enabled. The
  upstream qt-viewer-pro tests are intentionally disabled in this integrated
  configuration.

Complete local validation requires external inputs supplied through
`MAIW_CARDIAC_MRI_PACKAGE_DIR`, `MAIW_CARDIAC_MRI_ACDC_TRAINING_DIR`,
`MAIW_CARDIAC_MRI_REAL_ED_PATH`, and `MAIW_CARDIAC_MRI_REAL_ES_PATH`, together with
the authoritative Python repository and its dependencies for the independent parity
harness. Medical data and deployment artifacts are not stored in this repository.

## Deferred After Phase 9

The following work is explicitly non-blocking for Phase 9:

- stronger ED/ES patient and frame identity semantics;
- bounded worker shutdown or cancellation;
- real cardiac DICOM validation;
- qt-viewer-pro DICOM fixes;
- redesign of `MAIW_ENABLE_QTVIEWERPRO_INTEGRATION` dependency and packaging;
- install/export packaging;
- broader viewer and AI architecture work.
