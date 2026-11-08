"""Lithology interval extraction from scanned well-log images."""

__version__ = "0.1.0"


def _locate_tesseract() -> None:
    """Point pytesseract at TESSERACT_CMD, or at the standard Windows install when tesseract is not on PATH."""
    import os
    import shutil
    import sys

    cmd = os.environ.get("TESSERACT_CMD")
    if not cmd and sys.platform == "win32" and shutil.which("tesseract") is None:
        default = os.path.join(os.environ.get("PROGRAMFILES", r"C:\Program Files"), "Tesseract-OCR", "tesseract.exe")
        cmd = default if os.path.exists(default) else None
    if cmd:
        import pytesseract

        pytesseract.pytesseract.tesseract_cmd = cmd


_locate_tesseract()
