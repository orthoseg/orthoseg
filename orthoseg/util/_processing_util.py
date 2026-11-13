"""Module containing utilities regarding processes."""

import logging
import multiprocessing
import multiprocessing.context
import os
import shutil
import subprocess
from collections.abc import Callable
from concurrent import futures
from types import TracebackType
from typing import Any

import psutil

WORKER_TYPES = {"threads", "processes"}


class PooledExecutorFactory:
    """Context manager to create a pooled executor.

    Args:
        worker_type (str, optional): type of executor pool to create.
            "threads" for a ThreadPoolExecutor, "processes" for a ProcessPoolExecutor.
        max_workers (int, optional): Maximum number of workers.
            Defaults to None to get automatic determination.
        initializer (function, optional): Function that does initialisations.
            Defaults to None.
        initargs (tuple, optional): Arguments to pass to the initializer function.
            Defaults to an empty tuple.
        mp_context (BaseContext, optional): multiprocessing context if processes are
            used. If None, "forkserver" will be used on linux to avoid risks on getting
            deadlocks. Defaults to None.
        nice_value: The niceness value to set for the worker. 19 is the maximum niceness
            (lowest priority), and -20 is the minimum (highest priority). If None, the
            niceness will not be changed. Defaults to 15.
        openmp_threads: The size of the OpenMP threadpool to set for the worker.
            If None, the number of OpenMP threads is not changed.
            If 0, the pool size is determined as follows: for "threads" workers it is
            not changed, for "processes" it will be determined by the environment
            variable `GFO_WORKER_OMP_NUM_THREADS` or default to 1. Defaults to 0.
    """

    def __init__(
        self,
        worker_type: str = "processes",
        max_workers: int | None = None,
        initializer: Callable | None = None,
        initargs: tuple = (),
        mp_context: multiprocessing.context.BaseContext | None = None,
        *,
        nice_value: int | None = 15,
        openmp_threads: int | None = 0,
    ) -> None:
        self.worker_type = worker_type.lower()
        if self.worker_type not in WORKER_TYPES:
            raise ValueError(
                f"Invalid worker_type: {self.worker_type}. "
                f"Must be one of {WORKER_TYPES}."
            )

        self.max_workers = max_workers
        if max_workers is not None and os.name == "nt":
            # On windows, max workers should be limited to 61 to avoid errors
            self.max_workers = min(max_workers, 61)

        self.initializer, self.initargs = self._resolve_initializer_config(
            worker_type=self.worker_type,
            initializer=initializer,
            initargs=initargs,
            nice_value=nice_value,
            openmp_threads=openmp_threads,
        )

        self.mp_context = mp_context
        if mp_context is None and os.name not in {"nt", "darwin"}:
            # On linux, overrule default to "forkserver" to avoid risks to deadlocks
            self.mp_context = multiprocessing.get_context("forkserver")
        self.pool: futures.Executor | None = None

    def __enter__(self) -> futures.Executor:
        if self.worker_type == "threads":
            self.pool = futures.ThreadPoolExecutor(
                max_workers=self.max_workers,
                initializer=self.initializer,
                initargs=self.initargs,
            )
        elif self.worker_type == "processes":
            self.pool = futures.ProcessPoolExecutor(
                max_workers=self.max_workers,
                initializer=self.initializer,
                initargs=self.initargs,
                mp_context=self.mp_context,
            )
        else:
            raise ValueError(
                f"Invalid worker_type: {self.worker_type}. "
                f"Must be one of {WORKER_TYPES}."
            )

        return self.pool

    def __exit__(
        self,
        type: type,  # noqa: A002
        value: Exception | None,
        traceback: TracebackType | None,
    ) -> None:
        if self.pool is not None:
            self.pool.shutdown(wait=True)

    @staticmethod
    def _resolve_initializer_config(
        worker_type: str,
        initializer: Callable | None,
        initargs: tuple,
        nice_value: int | None,
        openmp_threads: int | None,
    ) -> tuple[Callable | None, tuple]:
        if initializer is None:
            if nice_value is not None and (nice_value < -20 or nice_value > 19):
                raise ValueError("nice_value must be between -20 and 19")
            if openmp_threads is not None and openmp_threads < 0:
                raise ValueError("openmp_threads must be non-negative")
            if initargs:
                raise ValueError("Cannot set initargs when using default initializer")
            if openmp_threads == 0 and worker_type.lower() == "threads":
                openmp_threads = None
            if nice_value is None and openmp_threads is None:
                return None, ()
            return initialize_worker, (nice_value, openmp_threads)

        if nice_value is not None or openmp_threads is not None:
            raise ValueError(
                "nice_value and openmp_threads must be None with a custom initializer"
            )
        return initializer, initargs


def initialize_worker(
    nice_value: int | None = 15, openmp_threads: int | None = None
) -> None:
    """Some default inits.

    Following things are done:
    - Set the worker process priority low (to `nice_value`) so the workers don't block
      the system.
    - Set the size of the OpenMP threadpool. This helps avoid committed memory getting
      very high when using multiprocessing.

    Args:
        nice_value: The niceness value to set for the worker. 19 is the maximum niceness
            (lowest priority), and -20 is the minimum (highest priority). If None, the
            niceness will not be changed. Defaults to 15.
        openmp_threads: The size of the OpenMP threadpool to set for the worker.
            If None, the number of OpenMP threads is not changed.
            If 0, the number of OpenMP threads will be determined by the environment
            variable `GFO_WORKER_OMP_NUM_THREADS` or default to 1.
    """
    # Set the number of OpenMP threads to avoid unnecessary inflated committed
    # memory when using multiprocessing.
    # Should work for any numeric library used (openblas, mkl,...).
    # Ref: https://stackoverflow.com/questions/77764228/pandas-scipy-high-commit-memory-usage-windows
    if openmp_threads is not None:
        if openmp_threads == 0:
            openmp_threads = int(os.environ.get("GFO_WORKER_OMP_NUM_THREADS", "1"))
        os.environ["OMP_NUM_THREADS"] = str(openmp_threads)

    # We don't want the workers to block the entire system, so make the workers nice
    # if they aren't quite nice already.
    #
    # Remarks:
    #   - on linux, depending on system settings it is not possible to decrease
    #     niceness, even if it was you who niced before.
    #   - we are setting the niceness of the process here, so in case of
    #     `worker_type=threads` the main thread/process will also be impacted.
    if nice_value is not None and getprocessnice() < nice_value:
        setprocessnice(nice_value)


def get_gpu_memory_info() -> list[dict[str, Any]]:
    """Get system-wide memory information for all available NVIDIA GPUs.

    Does not initialize the TensorFlow or CUDA runtime in the current process.

    Returns:
        List of dicts with keys: index, name, used_mb, total_mb, free_mb.
    """
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return []

    try:
        res = subprocess.run(
            [
                nvidia_smi,
                "--query-gpu=index,name,memory.used,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        gpus = []
        for line in res.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == 5:
                gpus.append(
                    {
                        "index": int(parts[0]),
                        "name": parts[1],
                        "used_mb": float(parts[2]),
                        "total_mb": float(parts[3]),
                        "free_mb": float(parts[4]),
                    }
                )
        return gpus
    except Exception:
        return []


def log_gpu_memory_info(logger: logging.Logger | None = None) -> None:
    """Log system-wide memory information for available NVIDIA GPUs.

    Args:
        logger: Optional logger to use. If None, the module logger is used.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    for gpu in get_gpu_memory_info():
        msg = (
            f"GPU {gpu['index']} ({gpu['name']}): "
            f"{gpu['used_mb']:.0f} MiB used / {gpu['total_mb']:.0f} MiB total "
            f"({gpu['free_mb']:.0f} MiB free)"
        )
        if gpu["free_mb"] < 2048:
            logger.warning(f"Low GPU memory available: {msg}")
        else:
            logger.info(f"GPU memory: {msg}")


def getprocessnice() -> int:
    """Get the niceness of the current process.

    The nice value can (typically) range from 19, which gives all other
    processes priority, to -20, which means that this process will take
    maximum priority (which isn't very nice ;-)).

    Remarks for windows:
        - windows only supports 6 niceness classes. setprocessnice en
          getprocessnice maps niceness values to these classes.
        - when setting REALTIME priority (-20 niceness) apparently this
          results only to HIGH priority.
    """
    p = psutil.Process(os.getpid())
    nice_value = p.nice()
    if os.name == "nt":
        return process_priorityclass_to_nice(nice_value)
    else:
        return int(nice_value)


def setprocessnice(nice_value: int) -> None:
    """Set the niceness of the current process.

    The nice value can (typically) range from 19, which gives all other
    processes priority, to -20, which means that this process will take
    maximum priority (which isn't very nice ;-)).

    Remarks for windows:
        - windows only supports 6 niceness classes. setprocessnice en
          getprocessnice maps niceness values to these classes.
        - when setting REALTIME priority (-20 niceness) apparently this
          results only to HIGH priority.

    Args:
        nice_value: the niceness to be set.
    """
    if nice_value < -20 or nice_value > 19:
        raise ValueError(
            f"Invalid value for nice_values (min: -20, max: 19): {nice_value}"
        )
    if getprocessnice() == nice_value:
        # If the nice value is already the same... no use setting it
        return

    try:
        p = psutil.Process(os.getpid())
        if os.name == "nt":
            p.nice(process_nice_to_priorityclass(nice_value))
        else:
            p.nice(nice_value)
    except Exception as ex:  # pragma: no cover
        raise RuntimeError(
            f"Error in setprocessnice with nice_value: {nice_value}"
        ) from ex


def process_nice_to_priorityclass(nice_value: int) -> int:  # pragma: no cover
    if nice_value == -20:
        return psutil.REALTIME_PRIORITY_CLASS
    elif nice_value <= -15:
        return psutil.HIGH_PRIORITY_CLASS
    elif nice_value <= -10:
        return psutil.ABOVE_NORMAL_PRIORITY_CLASS
    elif nice_value <= 0:
        return psutil.NORMAL_PRIORITY_CLASS
    elif nice_value <= 10:
        return psutil.BELOW_NORMAL_PRIORITY_CLASS
    else:
        return psutil.IDLE_PRIORITY_CLASS


def process_priorityclass_to_nice(priority_class: int) -> int:  # pragma: no cover
    if priority_class == psutil.REALTIME_PRIORITY_CLASS:
        return -20
    elif priority_class == psutil.HIGH_PRIORITY_CLASS:
        return -15
    elif priority_class == psutil.ABOVE_NORMAL_PRIORITY_CLASS:
        return -10
    elif priority_class == psutil.NORMAL_PRIORITY_CLASS:
        return 0
    elif priority_class == psutil.BELOW_NORMAL_PRIORITY_CLASS:
        return 10
    elif priority_class == psutil.IDLE_PRIORITY_CLASS:
        return 19
    else:
        return 0
