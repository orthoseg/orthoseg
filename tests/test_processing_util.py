"""Tests for module orthoseg.util._processing_util."""

import logging
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

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


@pytest.mark.parametrize(
    ("gpu", "expected"),
    [
        (
            {
                "index": 0,
                "name": "NVIDIA L4",
                "used_mb": 1000.0,
                "total_mb": 24000.0,
                "free_mb": 23000.0,
            },
            (
                "GPU memory: GPU 0 (NVIDIA L4): 1000 MiB used / 24000 MiB total "
                "(23000 MiB free)"
            ),
        ),
        (
            {
                "index": 0,
                "name": "NVIDIA L4",
                "used_mb": 23000.0,
                "total_mb": 24000.0,
                "free_mb": 1000.0,
            },
            (
                "Low GPU memory available: GPU 0 (NVIDIA L4): 23000 MiB used / "
                "24000 MiB total (1000 MiB free)"
            ),
        ),
    ],
)
def test_log_gpu_memory_info(monkeypatch, caplog, gpu, expected):
    monkeypatch.setattr(_processing_util, "get_gpu_memory_info", lambda: [gpu])

    with caplog.at_level(logging.INFO):
        _processing_util.log_gpu_memory_info()

    assert expected in caplog.text


def test_log_gpu_memory_info_no_gpu(monkeypatch, caplog):
    monkeypatch.setattr(_processing_util, "get_gpu_memory_info", list)

    with caplog.at_level(logging.INFO):
        _processing_util.log_gpu_memory_info()

    assert caplog.text == ""


def test_pooled_executor_factory_custom_initializer_config():
    initializer, initargs = (
        _processing_util.PooledExecutorFactory._resolve_initializer_config(
            worker_type="processes",
            initializer=len,
            initargs=("test",),
            nice_value=None,
            openmp_threads=None,
        )
    )

    assert initializer is len
    assert initargs == ("test",)


@pytest.mark.parametrize(
    ("worker_type", "expected_initargs"),
    [
        ("processes", (15, 0)),
        ("threads", (15, None)),
        ("THREADS", (15, None)),
    ],
)
def test_pooled_executor_factory_default_initializer_config(
    worker_type, expected_initargs
):
    initializer, initargs = (
        _processing_util.PooledExecutorFactory._resolve_initializer_config(
            worker_type=worker_type,
            initializer=None,
            initargs=(),
            nice_value=15,
            openmp_threads=0,
        )
    )

    assert initializer is _processing_util.initialize_worker
    assert initargs == expected_initargs


@pytest.mark.parametrize(
    "expected_start_method",
    [
        pytest.param(
            "forkserver",
            id="forkserver",
            marks=pytest.mark.skipif(
                os.name == "nt" or sys.platform == "darwin",
                reason="This platform does not select forkserver by default.",
            ),
        ),
        pytest.param(
            None,
            id="platform-default",
            marks=pytest.mark.skipif(
                os.name != "nt" and sys.platform != "darwin",
                reason="This platform selects forkserver by default.",
            ),
        ),
    ],
)
def test_pooled_executor_factory_default_mp_context(expected_start_method):
    factory = _processing_util.PooledExecutorFactory(worker_type="processes")

    actual_start_method = (
        factory.mp_context.get_start_method()
        if factory.mp_context is not None
        else None
    )
    assert actual_start_method == expected_start_method


def test_pooled_executor_factory_disables_initializer():
    initializer, initargs = (
        _processing_util.PooledExecutorFactory._resolve_initializer_config(
            worker_type="processes",
            initializer=None,
            initargs=(),
            nice_value=None,
            openmp_threads=None,
        )
    )

    assert initializer is None
    assert initargs == ()


@pytest.mark.parametrize(
    ("initializer", "initargs", "nice_value", "openmp_threads", "message"),
    [
        (None, (), -21, 0, "nice_value must be between -20 and 19"),
        (None, (), 20, 0, "nice_value must be between -20 and 19"),
        (None, (), 15, -1, "openmp_threads must be non-negative"),
        (None, ("test",), 15, 0, "Cannot set initargs"),
        (
            len,
            (),
            15,
            None,
            "nice_value and openmp_threads must be None with a custom initializer",
        ),
    ],
)
def test_pooled_executor_factory_invalid_initializer_config(
    initializer, initargs, nice_value, openmp_threads, message
):
    with pytest.raises(ValueError, match=message):
        _processing_util.PooledExecutorFactory._resolve_initializer_config(
            worker_type="processes",
            initializer=initializer,
            initargs=initargs,
            nice_value=nice_value,
            openmp_threads=openmp_threads,
        )


def test_pooled_executor_factory_invalid_worker_type():
    with pytest.raises(ValueError, match="Invalid worker_type"):
        _processing_util.PooledExecutorFactory(worker_type="invalid")


@pytest.mark.skipif(os.name != "nt", reason="The worker limit is specific to Windows.")
def test_pooled_executor_factory_limits_workers_on_windows():
    factory = _processing_util.PooledExecutorFactory(
        worker_type="processes", max_workers=100
    )

    assert factory.max_workers == 61


@pytest.mark.parametrize(
    ("worker_type", "executor_class", "max_workers", "mp_context", "expected_initargs"),
    [
        ("processes", "ProcessPoolExecutor", 3, object(), (15, 0)),
        ("THREADS", "ThreadPoolExecutor", 2, None, (15, None)),
    ],
    ids=["processes", "threads"],
)
def test_pooled_executor_factory_with_worker_type(
    monkeypatch,
    worker_type,
    executor_class,
    max_workers,
    mp_context,
    expected_initargs,
):
    shutdown_calls = []
    executor = SimpleNamespace(shutdown=lambda **kwargs: shutdown_calls.append(kwargs))
    executor_arguments = {}

    def create_executor(**kwargs):
        executor_arguments.update(kwargs)
        return executor

    monkeypatch.setattr(_processing_util.futures, executor_class, create_executor)

    factory = _processing_util.PooledExecutorFactory(
        worker_type=worker_type, max_workers=max_workers, mp_context=mp_context
    )
    assert factory.initializer is _processing_util.initialize_worker
    assert factory.initargs == expected_initargs

    with factory as pool:
        assert pool is executor

    expected_arguments = {
        "max_workers": max_workers,
        "initializer": _processing_util.initialize_worker,
        "initargs": expected_initargs,
    }
    if mp_context is not None:
        expected_arguments["mp_context"] = mp_context
    assert executor_arguments == expected_arguments
    assert shutdown_calls == [{"wait": True}]
