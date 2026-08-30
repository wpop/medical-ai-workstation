# Medical AI Workstation Architecture

## Architectural intent

Medical AI Workstation separates reusable medical imaging, workstation-specific
viewer composition, cardiac preprocessing, generic inference, application
services, and Qt presentation. Dependencies point inward toward data and
service contracts; presentation code does not own numerical algorithms.

The current application is composed as follows:

```text
Qt presentation
  |
  +-- ViewerWorkspaceWidget
  |       `-- external qt-viewer-pro core / IO / processing / render
  |
  `-- CardiacMriClassificationWorkflow
          +-- volume loading --> external qt-viewer-pro IO
          |
          `-- CardiacMriClassificationService
                  +-- CardiacMriPreprocessor --> external qt-viewer-pro core
                  |
                  `-- generic OnnxRuntimeSession --> ONNX Runtime
```

`src/app/main.cpp` is the composition root. It creates the external runtime
environment, service, asynchronous workflow, and top-level window in explicit
lifetime order.

## External qt-viewer-pro boundary

qt-viewer-pro is a separate, read-only repository integrated at build time with
`add_subdirectory` when `MAIW_ENABLE_QTVIEWERPRO_INTEGRATION=ON`. Medical AI
Workstation consumes its reusable libraries rather than its application UI:

- **core** owns medical volume data, slice extraction, anatomical orientation,
  metadata, and voxel-to-physical coordinate mapping;
- **IO** selects NIfTI, MetaImage, DICOM, NRRD, or RAW/JSON loaders and returns a
  common volume result;
- **processing** converts slice data and provides viewer image processing;
- **render** provides the OpenGL slice and 3D volume-rendering widgets.

The dependency has no current install/export package, so source integration is
an explicit build boundary. Its source is not copied into or modified by this
repository. Upstream tests are intentionally disabled in the parent CTest
registry because qt-viewer-pro registers them unconditionally during source
integration; MAIW integration tests exercise the consumed boundary.

## MAIW viewer layer

The MAIW viewer layer adapts the reusable qtvp facilities to the workstation.
It contains no cardiac model or inference policy.

### `VolumeLoadWorkflow`

`VolumeLoadWorkflow` validates empty and overlapping requests, then calls the
qtvp medical-volume registry inside `QtConcurrent::run`. A `QFutureWatcher`
delivers the result back to the workflow object's thread. Loading exceptions
and invalid volumes cross the Qt boundary as controlled failure signals.

The workflow destructor waits for an active future. This is a final ownership
barrier, not a cancellation mechanism.

### `ViewerWorkspaceWidget`

`ViewerWorkspaceWidget` owns:

- one `VolumeLoadWorkflow` through Qt parent ownership;
- one canonical `shared_ptr<const qvp::VolumeData>`;
- one `MprViewerWidget` observing that volume without ownership;
- one `VolumeRenderingWidget` sharing ownership of the same immutable volume.

Replacement clears child assignments before releasing the workspace's
canonical handle. This keeps the MPR non-owning pointer valid throughout each
transition and avoids a second voxel store.

### MPR and slice viewing

`MprViewerWidget` composes axial and sagittal views in its top row and a coronal
view across the lower row. One clamped voxel index is the navigation source of
truth. A user-selected image point updates the appropriate voxel axes, after
which all slice indices and crosshairs are synchronized.

`SliceViewerWidget` delegates slice extraction and OpenGL presentation to qtvp.
Crosshairs are stored in slice-image pixel coordinates and mapped to the fitted
image rectangle for presentation. qtvp orientation metadata and
`VolumePhysicalCoordinateMapper` provide the orientation-aware and physical
coordinate behavior.

### 3D rendering

`VolumeRenderingWidget` wraps the public qtvp OpenGL volume renderer. It shares
the workspace's immutable volume, exposes render presets, opacity and intensity
controls, forwards GPU upload results, and leaves GPU resource management to
the renderer.

## Cardiac classification

### `CardiacMriClassificationWorkflow`

The Qt workflow owns execution state and the asynchronous boundary. Before work
starts, it validates:

1. no classification is already running;
2. both ED and ES paths are non-empty;
3. existing ED and ES paths are not positively proven to reference the same
   filesystem object;
4. both file names use the validated, case-insensitive `.nii` or `.nii.gz`
   cardiac input contract.

It then emits `classificationStarted()` and launches one `QtConcurrent` worker.
The worker loads ED followed by ES through qtvp IO and calls the synchronous
classification service. The `QFutureWatcher` converts completion into success
or controlled failure signals on the workflow object's thread.

The workflow references, but does not own, the service. Its destructor waits
for active work so the service cannot be used after workflow destruction.

### `CardiacMriClassificationService`

`CardiacMriClassificationService` is a synchronous, headless application
service. It owns:

- validated deployment metadata;
- `CardiacMriPreprocessor`;
- the generic `OnnxRuntimeSession`.

Construction validates the deployed model's names, float32 element types,
ranks, and input/output dimensions against deployment metadata and the frozen
cardiac tensor contract. `classify()` preprocesses the pair, runs inference,
validates the returned logits shape, and creates the validated classification
result. The service has no Qt UI dependency.

### Cardiac preprocessing

The cardiac preprocessor is model-specific and remains separate from both the
generic viewer and ONNX Runtime wrapper. The current frozen pipeline:

```text
ED/ES qvp volumes
  -> trusted-orientation normalization to LPS
  -> ED/ES oriented geometry validation
  -> trilinear resampling on one ED-derived 1.5 x 1.5 x 7.5 grid
  -> shared 144 x 144 FOV-center crop
  -> joint 0.5/99.5 percentile clipping and z-score normalization
  -> center padding to depth 14
  -> XYZ-to-DHW conversion
  -> float32 CDHW tensor [2, 14, 144, 144]
```

Channel 0 is ED and channel 1 is ES. Geometry validation establishes the
implemented spatial compatibility contract; it does not establish patient,
study, or temporal-phase identity.

### Generic ONNX Runtime

`OnnxRuntimeSession` owns `Ort::SessionOptions`, `Ort::Session`, and inspected
tensor metadata. It validates generic tensor names, types, shapes, and element
counts before executing ONNX Runtime. It contains no cardiac labels, ED/ES
rules, softmax, diagnosis logic, or preprocessing policy.

## Qt presentation

Presentation remains separate from execution and numerical behavior:

- `CardiacMriClassificationWindow` owns ED/ES path controls, workflow control
  enablement, and transient progress/success/failure status;
- `CardiacMriClassificationResultWidget` owns predicted-class, probability, and
  controlled error presentation;
- `MedicalAiWorkstationWindow` composes the viewer and cardiac panel, forwards
  committed paths to the viewer, and enforces the top-level close policy.

No UI class loads volumes synchronously, preprocesses data, or invokes ONNX
Runtime directly.

## Composition and lifetime model

The composition root uses stack declaration order to establish the required
non-owning lifetime hierarchy:

```text
Ort::Env
  outlives
CardiacMriClassificationService
  outlives
CardiacMriClassificationWorkflow
  outlives
MedicalAiWorkstationWindow
```

Destruction occurs in reverse order. The window cannot outlive the workflow it
references; the workflow cannot outlive the service it invokes; and the service
cannot outlive the ONNX Runtime environment used by its session.

Qt parent-child ownership handles UI and viewer children. Shared ownership is
used only where the immutable viewer volume genuinely crosses the MPR/3D
ownership boundary.

## Concurrency, close, and shutdown

Both medical-volume loading and cardiac classification move blocking IO and
compute off the Qt event loop with `QtConcurrent`. Only one operation of each
workflow type may be active at a time.

The Phase 14 close policy rejects a normal top-level close event while either
classification or viewer loading is active. The event handler is non-blocking,
shows a transient status-bar message, and allows the operation to continue.
The user may close the workstation after the operation finishes.

There are no fake cancellation semantics. The application does not claim that
ITK IO or ONNX Runtime can be interrupted safely. As final lifetime barriers,
the `CardiacMriClassificationWorkflow` and `VolumeLoadWorkflow` destructors wait
for their active futures. This prevents dependent objects from being destroyed
while a worker still uses them, but it does not provide a fixed shutdown-time
guarantee for a worker blocked inside an external library.

## Runtime deployment

The install rules deploy:

- `medical-ai-workstation` to the install binary directory;
- Qt runtime dependencies through Qt's deployment script when using Qt 6.5 or
  newer;
- the ONNX Runtime library to the install library directory;
- `classifier.onnx`, `deployment.json`, and `class_mapping.json` to
  `share/medical-ai-workstation/cardiac_mri_pathology` when a package is
  configured.

At startup, `src/app/main.cpp` validates deployment metadata before creating the
service. The default package path is resolved relative to the installed
executable. `--cardiac-package` supplies an explicit override without changing
service or UI architecture.
