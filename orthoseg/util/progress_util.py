# -*- coding: utf-8 -*-
"""
Module with functions to manage progress logging.
"""

import datetime
import logging
from typing import Optional

# Get a logger...
logger = logging.getLogger(__name__)


class ProgressLogger:
    first_reporting_done = False

    def __init__(
        self,
        message: str,
        nb_steps_total: int,
        nb_steps_done: int = 0,
        start_time: Optional[datetime.datetime] = None,
        time_between_reporting_s: int = 60,
        calculate_eta_since_lastreporting: bool = True,
    ):
        self.message = message
        if start_time is None:
            self.start_time = datetime.datetime.now()
        else:
            self.start_time = start_time
        self.start_time_lastreporting = self.start_time
        self.nb_steps_total = nb_steps_total
        self.nb_steps_done = nb_steps_done
        self.nb_steps_done_lastreporting = nb_steps_done
        self.time_between_reporting_s = time_between_reporting_s
        self.calculate_eta_since_lastreporting = calculate_eta_since_lastreporting

    def step(self, message: Optional[str] = None, nb_steps: int = 1):

        # Increase done counter
        self.nb_steps_done += nb_steps

        # Calculate time since last reporting
        time_now = datetime.datetime.now()
        time_passed_lastprogress_s = (
            time_now - self.start_time_lastreporting
        ).total_seconds()

        # Calculate the time_passed and nb_done we want to use to calculate ETA
        if self.calculate_eta_since_lastreporting is True:
            time_passed_for_eta_s = (
                time_now - self.start_time_lastreporting
            ).total_seconds()
            nb_steps_done_eta = self.nb_steps_done - self.nb_steps_done_lastreporting
        else:
            time_passed_for_eta_s = (time_now - self.start_time).total_seconds()
            nb_steps_done_eta = self.nb_steps_done

        # Print progress on first step, if sufficient time between reporting has passed
        # or if sufficient progress since last reporting
        pct_progress_since_last_reporting = (
            self.nb_steps_done - self.nb_steps_done_lastreporting
        ) / self.nb_steps_total
        if (
            time_passed_lastprogress_s >= self.time_between_reporting_s
            or pct_progress_since_last_reporting > 0.1
            or self.first_reporting_done is False
        ):
            # Evade divisions by zero
            if time_passed_for_eta_s == 0 or nb_steps_done_eta == 0:
                return

            nb_per_hour = (nb_steps_done_eta / time_passed_for_eta_s) * 3600
            hours_to_go = (int)(
                (self.nb_steps_total - self.nb_steps_done) / nb_per_hour
            )
            min_to_go = (int)(
                (((self.nb_steps_total - self.nb_steps_done) / nb_per_hour) % 1) * 60
            )
            if message is not None:
                progress_message = (
                    f"{message}, {self.nb_steps_done}/{self.nb_steps_total}, "
                    f"{hours_to_go:3d}:{min_to_go:2d} left at {nb_per_hour:0.0f}/h"
                )
            else:
                progress_message = (
                    f"{self.message}, {self.nb_steps_done}/{self.nb_steps_total}, "
                    f"{hours_to_go:3d}:{min_to_go:2d} left at {nb_per_hour:0.0f}/h"
                )
            logger.info(progress_message)

            self.start_time_lastreporting = time_now
            self.nb_steps_done_lastreporting = self.nb_steps_done
            if self.first_reporting_done is False:
                self.first_reporting_done = True
