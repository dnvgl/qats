# 0001 – PySide6 is the tested Qt binding; PyQt6 is best effort

- **Status:** Accepted
- **Date:** 2026-10-02
- **Release:** 5.4 (Installable again)

## Context

QATS accesses Qt through [qtpy](https://github.com/spyder-ide/qtpy), so it can run on either PySide6 or PyQt6. Since 5.0.0, PySide6 is installed with `qats` and is the default; users can install PyQt6 and select it with `QT_API=pyqt6`.

When QATS was started, PyQt was considered the more mature and stable binding. That was true in the Qt4/PySide1 era. Since Qt 6, PySide6 has been the official Qt Company binding and is released alongside each Qt version. The two are now at parity in features and stability.

## Gains and pains

| | PySide6 | PyQt6 |
|---|---|---|
| **Maintainer** | The Qt Company (official binding) | Riverbank Computing |
| **Features and stability** | At parity with PyQt6 since Qt 6 | At parity with PySide6 |
| **Licence** | LGPL: fits MIT-licensed QATS and lets us bundle it in a standalone installer | GPL or commercial: bundling it in an installer would impose GPL terms on the bundle |
| **Cost to support** | Already the default and installed with QATS | Supporting it through qtpy costs little code, but full testing doubles the GUI test matrix |

## Decision

- Keep qtpy, so PyQt6 keeps working for users who prefer it, on a best-effort basis.
- Run CI only with PySide6 (`QT_API=pyside6`).
- The standalone Windows installer planned for 6.2 bundles PySide6 only.
- Bug reports that only reproduce on PyQt6 are welcome, but fixes are best effort.

## Consequences

- Smaller CI matrix and one licence story for distribution.
- A PyQt6-specific regression can slip through CI unnoticed. Mention the best-effort status in the README installation section.
- GUI code must use the qtpy imports only, never `PySide6` or `PyQt6` directly.
