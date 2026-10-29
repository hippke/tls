"""OpenMP flags for the C kernel: unchanged off macOS, libomp for Apple clang."""

import os
import subprocess
import sys

from transitleastsquares.backends import c_kernel


def test_non_darwin_keeps_plain_fopenmp(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    assert c_kernel._openmp_flags("cc") == ["-fopenmp"]


def test_apple_clang_uses_libomp(monkeypatch, tmp_path):
    (tmp_path / "include").mkdir()
    (tmp_path / "include" / "omp.h").write_text("")
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setenv("LIBOMP_PREFIX", str(tmp_path))
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(a, 0, "Apple clang version 17", ""),
    )
    flags = c_kernel._openmp_flags("cc")
    assert flags[:2] == ["-Xpreprocessor", "-fopenmp"]
    assert "-lomp" in flags
    assert "-I" + os.path.join(str(tmp_path), "include") in flags
