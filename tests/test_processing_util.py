"""Tests for module orthoseg.util._processing_util."""

import logging
import subprocess
from types import SimpleNamespace

from orthoseg.util import _processing_util


def test_get_gpu_memory_info_handles_subprocess_error(monkeypatch):
    monkeypatch.setattr(
        _processing_util.shutil, "which", lambda _cmd: "/usr/bin/nvidia-smi"
    )

    def _mock_run(*_args, **_kwargs):
        raise subprocess.SubprocessError("error")

    monkeypatch.setattr(_processing_util.subprocess, "run", _mock_run)

    assert _processing_util.get_gpu_memory_info() == []


def test_get_gpu_memory_info_no_nvidia_smi(monkeypatch):
    monkeypatch.setattr(_processing_util.shutil, "which", lambda _cmd: None)

    assert _processing_util.get_gpu_memory_info() == []


def test_get_gpu_memory_info_success(monkeypatch):
    monkeypatch.setattr(
        _processing_util.shutil, "which", lambda _cmd: "/usr/bin/nvidia-smi"
    )

    stdout = (
        "0, NVIDIA L4, 1000, 24000, 23000\n"
        "1, NVIDIA RTX 4090, 23500, 24000, 500\n"
        "invalid line\n"
    )

    def _mock_run(*_args, **_kwargs):
        return SimpleNamespace(stdout=stdout)

    monkeypatch.setattr(_processing_util.subprocess, "run", _mock_run)

    gpus = _processing_util.get_gpu_memory_info()
    assert len(gpus) == 2
    assert gpus[0] == {
        "index": 0,
        "name": "NVIDIA L4",
        "used_mb": 1000.0,
        "total_mb": 24000.0,
        "free_mb": 23000.0,
    }
    assert gpus[1] == {
        "index": 1,
        "name": "NVIDIA RTX 4090",
        "used_mb": 23500.0,
        "total_mb": 24000.0,
        "free_mb": 500.0,
    }


def test_log_gpu_memory_info_high_memory(monkeypatch, caplog):
    gpus = [
        {
            "index": 0,
            "name": "NVIDIA L4",
            "used_mb": 1000.0,
            "total_mb": 24000.0,
            "free_mb": 23000.0,
        }
    ]
    monkeypatch.setattr(_processing_util, "get_gpu_memory_info", lambda: gpus)

    with caplog.at_level(logging.INFO):
        _processing_util.log_gpu_memory_info()

    expected = (
        "GPU memory: GPU 0 (NVIDIA L4): 1000 MiB used / 24000 MiB total "
        "(23000 MiB free)"
    )
    assert expected in caplog.text


def test_log_gpu_memory_info_low_memory(monkeypatch, caplog):
    gpus = [
        {
            "index": 0,
            "name": "NVIDIA L4",
            "used_mb": 23000.0,
            "total_mb": 24000.0,
            "free_mb": 1000.0,
        }
    ]
    monkeypatch.setattr(_processing_util, "get_gpu_memory_info", lambda: gpus)

    with caplog.at_level(logging.INFO):
        _processing_util.log_gpu_memory_info()

    expected = (
        "Low GPU memory available: GPU 0 (NVIDIA L4): 23000 MiB used / "
        "24000 MiB total (1000 MiB free)"
    )
    assert expected in caplog.text


def test_log_gpu_memory_info_no_gpu(monkeypatch, caplog):
    monkeypatch.setattr(_processing_util, "get_gpu_memory_info", list)

    with caplog.at_level(logging.INFO):
        _processing_util.log_gpu_memory_info()

    assert caplog.text == ""
