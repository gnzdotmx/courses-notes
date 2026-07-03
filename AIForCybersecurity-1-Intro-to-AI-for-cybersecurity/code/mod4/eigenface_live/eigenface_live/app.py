"""Qt application bootstrap."""

from __future__ import annotations

import platform
import sys

from PyQt6.QtWidgets import QApplication, QMessageBox

from eigenface_live.config import DEFAULT_CONFIG
from eigenface_live.ui.main_window import MainWindow


def run() -> int:
    if platform.system() != "Darwin":
        print("Eigenface Live supports macOS only.", file=sys.stderr)
        return 1

    app = QApplication(sys.argv)
    app.setApplicationName("Eigenface Live")
    app.setOrganizationName("AI for Cybersecurity")

    try:
        window = MainWindow(DEFAULT_CONFIG)
    except Exception as exc:
        QMessageBox.critical(None, "Startup error", str(exc))
        return 1

    window.show()
    return app.exec()
