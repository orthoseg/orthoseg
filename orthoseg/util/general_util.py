"""Module containing some general utilities."""

import datetime
import logging

logger = logging.getLogger(__name__)


class MissingRuntimeDependencyError(Exception):
    """Exception raised when an unsupported SQL statement is passed.

    Attributes:
        message (str): Exception message
    """

    def __init__(self, message):
        """Constructor of MissingRuntimeDependencyError.

        Args:
            message (str): message.
        """
        self.message = message
        super().__init__(self.message)


################################################################################
# The real work
################################################################################


def report_progress(
    start_time: datetime.datetime,
    nb_done: int,
    nb_todo: int,
    operation: str | None = None,
    nb_parallel: int = 1,
):
    """Function to report progress to the output.

    Args:
        start_time (datetime): time when the processing started.
        nb_done (int): number of steps done.
        nb_todo (int): total number of steps to do.
        operation (Optional[str], optional): operation being done. Defaults to None.
        nb_parallel (int, optional): number of parallel workers doing the processing.
            Defaults to 1.
    """
    # Init
    time_passed = (datetime.datetime.now() - start_time).total_seconds()
    pct_progress = 100.0 - (nb_todo - nb_done) * 100 / nb_todo

    # If we haven't really started yet, don't report time estimate yet
    if nb_done == 0:
        message = (
            f"\r  ?: ? left to do {operation} on {(nb_todo - nb_done):8d} "
            f"of {nb_todo:8d} ({pct_progress:3.2f}%)    "
        )
        print(message, end="", flush=True)
    elif time_passed > 0:
        # Else, report progress properly...
        processed_per_hour = (nb_done / time_passed) * 3600
        # Correct the nb processed per hour if running parallel
        if nb_done < nb_parallel:
            processed_per_hour = round(processed_per_hour * nb_parallel / nb_done)
        hours_to_go = (int)((nb_todo - nb_done) / processed_per_hour)
        min_to_go = (int)((((nb_todo - nb_done) / processed_per_hour) % 1) * 60)
        pct_progress = 100.0 - (nb_todo - nb_done) * 100 / nb_todo
        if pct_progress < 100:
            message = (
                f"\r{hours_to_go:3d}:{min_to_go:2d} left to do {operation} on "
                f"{(nb_todo - nb_done):8d} of {nb_todo:8d} ({pct_progress:3.2f}%)    "
            )
        else:
            message = (
                f"\r{hours_to_go:3d}:{min_to_go:2d} left to do {operation} on "
                f"{(nb_todo - nb_done):8d} of {nb_todo:8d} ({pct_progress:3.2f}%)    \n"
            )
        print(message, end="", flush=True)


def formatbytes(nb_bytes: float) -> str:
    """Return the given bytes as a human friendly KB, MB, GB, or TB string.

    Args:
        nb_bytes (float): number of bytes to format.

    Returns:
        str: number of bytes as a readable sting.
    """
    bytes_float = float(nb_bytes)
    KB = float(1024)
    MB = float(KB**2)  # 1,048,576
    GB = float(KB**3)  # 1,073,741,824
    TB = float(KB**4)  # 1,099,511,627,776

    if bytes_float < KB:
        return "{} {}".format(bytes_float, "Bytes" if bytes_float > 1 else "Byte")
    elif KB <= bytes_float < MB:
        return f"{bytes_float / KB:.2f} KB"
    elif MB <= bytes_float < GB:
        return f"{bytes_float / MB:.2f} MB"
    elif GB <= bytes_float < TB:
        return f"{bytes_float / GB:.2f} GB"
    else:
        return f"{bytes_float / TB:.2f} TB"
