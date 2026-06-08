import sys
import argparse
import faulthandler
import traceback

from PySide6.QtWidgets import QApplication, QMessageBox
from PySide6.QtCore import Qt

from ui.window import DitherGuy
from utils.constants import VERSION

def main() -> None:
    parser = argparse.ArgumentParser(description="Dither Guy")
    parser.add_argument("--debug", action="store_true", help="Enable developer debug mode")
    args = parser.parse_args()

    if args.debug:
        faulthandler.enable()
        print("Debug mode enabled. Faulthandler active.", file=sys.stderr)

    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    app = QApplication(sys.argv)
    app.setApplicationName("Dither Guy")
    app.setApplicationVersion(VERSION)

    if args.debug:
        def debug_excepthook(exc_type, exc_value, exc_tb):
            tb_str = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
            print("UNHANDLED EXCEPTION:\n" + tb_str, file=sys.stderr)
            try:
                QMessageBox.critical(None, "Fatal Debug Error", f"App crashed:\n\n{exc_value}\n\nCheck terminal for full stacktrace.")
            except Exception:
                pass
            sys.exit(1)
        sys.excepthook = debug_excepthook

    w = DitherGuy()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass
    main()
