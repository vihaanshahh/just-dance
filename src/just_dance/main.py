"""Main entry point for Just Dance Video Processor."""

import sys
from PySide6.QtWidgets import QApplication
from just_dance.gui.main_window import MainWindow


def main():
    """Launch the Just Dance Video Processor application."""
    app = QApplication(sys.argv)
    app.setApplicationName("Just Dance Video Processor")
    app.setApplicationVersion("0.1.0")

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
