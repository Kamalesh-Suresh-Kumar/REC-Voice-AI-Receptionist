from __future__ import annotations

import os
from pathlib import Path


def configure_cuda_dll_paths() -> None:
    """
    Add NVIDIA CUDA DLL directories installed inside
    the project's Python virtual environment to PATH.

    Required by faster-whisper / CTranslate2 on Windows.
    """

    # Project root:
    # REC-Voice-AI-Receptionist/
    project_root = Path(__file__).resolve().parents[3]

    venv = project_root / ".venv"

    cuda_directories = [
        venv / "Lib" / "site-packages" / "nvidia" / "cublas" / "bin",
        venv / "Lib" / "site-packages" / "nvidia" / "cudnn" / "bin",
        venv / "Lib" / "site-packages" / "nvidia" / "cuda_nvrtc" / "bin",
    ]

    existing_path = os.environ.get("PATH", "")

    paths_to_add = []

    for directory in cuda_directories:
        if directory.exists():
            directory_string = str(directory)

            if directory_string not in existing_path:
                paths_to_add.append(directory_string)

    if paths_to_add:
        os.environ["PATH"] = (
            ";".join(paths_to_add)
            + ";"
            + existing_path
        )

        # Python 3.8+ on Windows may use explicit DLL
        # directory registration.
        for directory in paths_to_add:
            try:
                os.add_dll_directory(directory)
            except (AttributeError, FileNotFoundError):
                pass

        print("[CUDA] NVIDIA DLL paths configured.")