# Known Limitations

This document distinguishes behavior supported by current validation evidence
from known engineering limitations and work intentionally deferred beyond the
v1 scope. It does not define new product requirements.

## Validated behavior

- Cardiac AI input is restricted to NIfTI files with case-insensitive `.nii` or
  `.nii.gz` suffixes.
- The workflow rejects ED and ES paths when the filesystem positively proves
  that both refer to the same existing object.
- Cardiac preprocessing normalizes trusted orientation to LPS and validates the
  implemented ED/ES geometry compatibility contract before resampling.
- Python/C++ preprocessing, service, end-to-end, and ONNX parity are covered by
  external ACDC-derived golden validation.
- Normal workstation close requests are rejected while viewer loading or
  classification is active, preserving the established object-lifetime order.

These statements describe engineering behavior for the tested reference
pipeline. They are not claims of clinical effectiveness.

## Known limitations

### 1. Cardiac AI format scope

The cardiac AI workflow is validated only for the NIfTI family:

- `.nii`
- `.nii.gz`

Case-insensitive extension matching is a format gate, not file-content
validation. A supported suffix can still reach a controlled loader error.

### 2. Viewer formats are not cardiac AI formats

The generic viewer can use qt-viewer-pro loaders for NIfTI, DICOM (`.dcm`),
MetaImage (`.mhd`, `.mha`), NRRD (`.nrrd`, `.nhdr`), and RAW data described by a
JSON sidecar. This broader viewer support does not imply cardiac AI validation
for DICOM, MetaImage, NRRD, RAW, or JSON inputs.

The workstation file filter currently advertises `.dicom`, while the qtvp DICOM
registry recognizes `.dcm`. The picker text is not an authoritative statement
of loader support.

### 3. Real cardiac DICOM inference is not validated

A real cardiac DICOM ED/ES inference workflow has not been validated. Existing
DICOM loading capability and source-level integration tests are not substitutes
for real paired cardiac inference evidence.

### 4. Geometry is not identity or phase verification

For NIfTI inputs, successful geometry compatibility validation does not
independently establish:

- patient identity;
- study identity;
- ED/ES semantic correctness;
- temporal ordering;
- swapped-phase detection.

The workflow does not claim verification through DICOM `PatientID`,
`StudyInstanceUID`, filenames, or another provenance source.

### 5. Distinct files can still form an incorrect pair

Selecting the same physical ED/ES file is rejected when
`std::filesystem::equivalent` positively identifies it. Different files that
are geometry-compatible can still belong to an incorrect patient, study, or
phase pairing. Filesystem identity rejection does not solve provenance.

### 6. Blocking work has no true cancellation contract

Medical IO, ITK processing, preprocessing, and ONNX Runtime inference execute
away from the Qt event loop, but the application does not claim that work
already blocked inside those libraries can be cancelled safely or promptly.
No cancellation state is presented to the user.

### 7. Close safety does not bound worker duration

Normal workstation close is rejected while asynchronous work is active. This
preserves lifetime safety and allows the operation to finish normally. Workflow
destructors also wait as final lifetime barriers.

There is no hard guarantee that an underlying blocked worker will terminate
within a fixed number of seconds. Waiting for completion is not equivalent to
cancellation or a bounded external-library shutdown.

### 8. qt-viewer-pro is a source dependency

qt-viewer-pro remains a build-time source integration dependency and is treated
as read-only. It does not currently provide the install/export package consumed
by this project, and its tests are intentionally disabled in the integrated
parent CTest registry.

### 9. Medical data and deployment artifacts are external

The cardiac model, ACDC medical dataset, deployment metadata, class mapping,
and golden artifacts are intentionally external. They are not committed to this
repository. Consequently, real-data, runtime-package, and parity tests are
conditional on explicitly configured external paths.

### 10. Engineering parity is not clinical generalization

Agreement with the Python reference pipeline and deployment ONNX Runtime on the
validated ACDC-derived cohort establishes engineering parity for those tested
inputs. It does not demonstrate model generalization to other scanners,
protocols, institutions, populations, pathologies, or clinical workflows.

The project has no regulatory approval, clinical certification, diagnostic
approval, or claim of production clinical deployment.

## Intentionally deferred beyond v1

The following areas remain outside the current v1 scope:

- independent patient, study, and temporal-phase provenance verification;
- real cardiac DICOM ED/ES classification validation;
- true cancellation semantics for work executing inside ITK or ONNX Runtime;
- replacement of the read-only qt-viewer-pro source integration with an
  installed/exported dependency package;
- clinical generalization, clinical performance evaluation, and regulatory
  qualification.

These boundaries are documented to prevent broader interpretations of the
implemented and validated engineering behavior.
