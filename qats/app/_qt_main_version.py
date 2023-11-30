from qtpy import API_NAME

if API_NAME in ("PySide2", "PyQt5"):
    QT_MAIN_VERSION = 5
elif API_NAME in ("PySide6", "PyQt6"):
    QT_MAIN_VERSION = 6
else:
    raise NotImplementedError(f"QT API not implemented: '{API_NAME}'")

