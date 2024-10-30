"""Module containing some general utilities."""

import os

import psutil


def process_nice_to_priority_class(nice_value: int) -> int:
    """Convert a linux nice value to a windows priority class.

    Args:
        nice_value (int): nice value between -20 and 20.

    Returns:
        int: windows priority class.
    """
    if nice_value <= -15:
        return psutil.REALTIME_PRIORITY_CLASS
    elif nice_value <= -10:
        return psutil.HIGH_PRIORITY_CLASS
    elif nice_value <= -5:
        return psutil.ABOVE_NORMAL_PRIORITY_CLASS
    elif nice_value <= 0:
        return psutil.NORMAL_PRIORITY_CLASS
    elif nice_value <= 10:
        return psutil.BELOW_NORMAL_PRIORITY_CLASS
    else:
        return psutil.IDLE_PRIORITY_CLASS


def setprocessnice(nice_value: int):
    """Make the process nicer to other processes.

    Args:
        nice_value (int): Value between -20 (highest priority) and 20 (lowest priority)
    """
    p = psutil.Process(os.getpid())
    if os.name == "nt":
        p.nice(process_nice_to_priority_class(nice_value))
    else:
        p.nice(nice_value)
