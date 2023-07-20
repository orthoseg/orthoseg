# -*- coding: utf-8 -*-
"""
Delete a directory faster (in windows)
"""

import shutil
import datetime as dt

del_dir = "X:/Monitoring/OrthoSeg/trees"

time_start = dt.datetime.now()
print(f"Start delete at {time_start}")
shutil.rmtree(del_dir)
time_taken = (dt.datetime.now() - time_start).total_seconds() / 3600
print(f"Time taken: {time_taken:.5f hours}")
