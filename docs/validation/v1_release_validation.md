# Medical AI Workstation v1 Release Validation

## Release Candidate

- Version: 1.0.0
- Branch: `phase/15-final-release-validation`
- Automated-validation HEAD: `e5b1e0e8ac18ccee43c261853cfaf9d4566f5f73`

Documentation, version, or CI commits may follow this automated-validation
baseline before the final merge and tag. The final Git identity must be recorded
after the release Pull Request is merged.

## Automated Release Validation

- Fresh Release/Ninja configure: **PASS**
- Full build: **118/118 steps PASS**
- Project-owned compiler warnings: **none**
- Active MAIW/integration CTests: **25/25 PASS**
- Disabled upstream qt-viewer-pro registrations: **17**, intentionally disabled

The explicit release gates passed:

- `cardiac-mri-preprocessing-golden-parity` — **PASS**
- `cardiac-mri-end-to-end-golden-parity` — **PASS**
- `cardiac-mri-classification-service-golden-parity` — **PASS**
- `cardiac-mri-onnx-golden-parity` — **PASS**
- `cardiac-mri-classification-workflow` — **PASS**
- `cardiac-mri-classification-shutdown` — **PASS**
- `medical-ai-workstation-window` — **PASS**
- `medical-ai-workstation-lifecycle` — **PASS**
- `medical-ai-workstation-study-coordination` — **PASS**
- `slice-viewer-widget-real-volume` — **PASS**
- `mpr-viewer-widget-real-volume` — **PASS**
- `volume-rendering-widget-real-volume` — **PASS**
- `viewer-workspace-widget-real-volume` — **PASS**

## Five-Patient Non-Golden ACDC Parity

The existing independent parity harness validated the fixed non-golden cohort:

| Patient | Predicted class | Maximum logit error | Maximum probability error |
| --- | --- | ---: | ---: |
| patient002 | DCM | 0.0 | 0.0 |
| patient022 | HCM | 0.0 | 0.0 |
| patient041 | MINF | 0.0 | 0.0 |
| patient061 | NOR | 0.0 | 0.0 |
| patient081 | RV | 0.0 | 0.0 |

- Logit parity: **PASS**
- Probability parity: **PASS**
- Predicted-class parity: **PASS**
- Cohort result: **5/5 PASS**

Engineering parity demonstrates agreement between the validated Python and C++
execution paths for these studies. It is not evidence of clinical
generalization.

## Installed Runtime Validation

- Fresh isolated install: **PASS**
- Installed executable present and executable: **PASS**
- Installed cardiac package assets present: **PASS**
- Installed version: `Medical AI Workstation 1.0.0`
- Installed version exit code: `0`
- RUNPATH: `$ORIGIN/../lib`
- Source-tree or build-tree paths in dynamic runtime metadata: **none**
- `ldd` dependencies marked `not found`: **none**
- ONNX Runtime resolves from the staged install tree.
- Deployed Qt libraries resolve from the staged install tree.
- Normal system libraries remain system dependencies.

## Manual Desktop Acceptance

The user performed final manual validation on the real desktop after automated
validation.

Result: **PASS**

Verified with real patient002 ED/ES data:

- ED volume loaded successfully in the viewer.
- Axial, sagittal, and coronal MPR displayed correctly.
- Crosshair synchronization worked.
- 3D volume rendering displayed.
- Cardiac MRI classification completed successfully.
- The predicted result displayed as DCM.
- The application closed normally after work completed.

## Safety and Scope Confirmation

Final validation preserved:

- the Phase 12 same-physical-file ED/ES rejection;
- the Phase 13 NIfTI-only cardiac AI input boundary;
- the Phase 14 lifecycle-safe close behavior;
- validated cardiac preprocessing and inference semantics;
- generic viewer format behavior;
- the read-only qt-viewer-pro boundary.

## External Environment Notes

- The system GDCM package emitted warnings about missing command-line
  executables; configure and build were unaffected.
- `ctest -N` reported missing executables only for intentionally disabled
  upstream qt-viewer-pro tests.
- These messages were not active MAIW test failures.

## Release Assessment

- Automated release blockers: **none**
- Manual desktop blockers: **none**
- v1 engineering release criteria: **satisfied, subject to final Git, Pull
  Request, and CI completion**
- Regulatory or clinical certification claim: **none**

Phase 15 is not complete until the Pull Request is merged into `main`. No v1 tag
or completed PR/CI merge is claimed by this validation record.
