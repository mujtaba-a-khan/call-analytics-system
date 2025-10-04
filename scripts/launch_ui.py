"""Helper script to launch the Streamlit UI from the CLI."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
    """Launch the Streamlit application for the call analytics UI."""

    project_root = Path(__file__).resolve().parent.parent
    app_path = project_root / "src" / "ui" / "app.py"

    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        *sys.argv[1:],
    ]

    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
