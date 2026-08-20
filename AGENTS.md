# Medical AI Workstation — Codex Rules

## Language

* Explain all work and decisions to the user in Russian.
* Use professional English for code, identifiers, filenames, comments, documentation, CMake, tests, commits, and Pull Requests.
* Never write Russian comments in source code.

## File Editing

When giving the user manual editing commands:

* new file → `nano <exact-path>`
* existing file → `code <exact-path>`

Codex may use its own editing tools internally.

## C++ Quality

Use C++20 and professional engineering practices:

* RAII;
* const correctness;
* explicit ownership and lifetime;
* exception safety;
* small focused interfaces;
* target-based CMake;
* testable non-UI code.

Prefer `std::unique_ptr` for exclusive ownership.

Use `std::shared_ptr` only when shared ownership is genuinely required.

Never use raw owning pointers.

Avoid hidden global state and unnecessary Singletons.

For project-owned GCC/Clang targets use:

```text
-Wall
-Wextra
-Wpedantic
```

Do not use `-Werror` unless explicitly requested.

## OOP and Architecture

Use OOP where it provides a real architectural or domain benefit.

Follow SOLID principles, but do not apply them mechanically.

Prefer composition over inheritance.

Do not create unnecessary interfaces, wrappers, inheritance hierarchies, micro-libraries, or service classes.

Use GoF patterns only when they solve a real problem. Name patterns accurately.

Keep these responsibilities separate:

* medical image core and IO;
* viewer processing;
* task-specific AI preprocessing;
* generic ONNX Runtime;
* application services;
* Qt UI.

Do not put ONNX Runtime in `MainWindow`.

Do not put cardiac preprocessing in generic viewer or runtime classes.

Do not create speculative abstractions for future AI modalities.

## Documentation

Provide professional English documentation comments for every production class and every public production method.

Document non-obvious private methods when needed.

Comments should explain contracts, ownership, invariants, numerical assumptions, or architectural intent.

Do not add comments that merely repeat the code.

## Upstream Repositories

Treat these repositories as read-only unless the user explicitly approves changes:

```text
/mnt/4086152D86152546/MedProjects/qt-viewer-pro
/mnt/4086152D86152546/CardiacMedProjects/cardiac-mri-pathology-classification
```

Active repository:

```text
/mnt/4086152D86152546/MedProjects/medical-ai-workstation
```

Do not copy upstream source code into this repository without explicit justification.

## Medical AI Validation

Numerical Python ↔ C++ parity is a release gate.

Use real ACDC-derived golden references for medical AI validation.

Do not use synthetic medical images as medical validation.

Synthetic arrays are allowed only for pure utility or mathematical unit tests.

Never loosen tolerances merely to make tests pass. Find the numerical root cause first.

Do not copy model or golden artifacts into this repository unless explicitly approved.

## ONNX Runtime

Generic ONNX Runtime code must remain model- and modality-independent.

It must not contain cardiac labels, ED/ES logic, softmax, diagnosis logic, or preprocessing rules.

Use explicit RAII ownership.

Do not use a global `Ort::Env` Singleton.

Prefer CPU inference until correctness and parity are established.

## Qt

Do not block the Qt event loop with preprocessing or inference.

Keep application services headless and testable.

Keep Qt-specific asynchronous execution outside the core inference services.

## Workflow

Work one logical milestone at a time.

Do not implement future phases prematurely.

Always explicitly report:

```text
Phase N started
Phase N ready for Pull Request
Phase N completed
```

A Phase is completed only after its Pull Request is merged into `main`.

Do not silently move to the next Phase.

## Branches and Pull Requests

Each Phase must have:

* its own Git branch;
* its own Pull Request into `main`.

Example:

```text
phase/03-qt-viewer-integration
phase/04-cardiac-preprocessing
```

Do not implement a new Phase directly on `main`.

Do not start the next Phase until the current Phase is merged unless the user explicitly approves parallel work.

Pull Request titles and descriptions must use professional English.

Do not merge Pull Requests unless the user explicitly requests it.

## Git

Do not use `git add .` when explicit staging is practical.

Do not stage or commit unless explicitly requested.

Before commit or Pull Request verify:

```bash
git diff --check
git diff --cached --check
git status --short
```

Do not commit build files, IDE state, private medical data, generated golden data, or large model artifacts unless explicitly approved.

## Completion

Before declaring implementation successful, verify:

* semantic correctness;
* ownership and lifetime;
* build;
* compiler warnings;
* relevant tests;
* parity tests when applicable;
* Git status.

Report results in Russian.

Never claim a validation passed unless it was actually run.
