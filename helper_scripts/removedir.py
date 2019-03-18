# -*- coding: utf-8 -*-
"""
Created on Sat Jan 12 13:05:32 2019

@author: Pieter Roggemans
"""

import shutil
import datetime as dt

# If the script is ran directly...
if __name__ == '__main__':
    
    # Voorspelling in windows: 
    #    * 1 uur voorspellen, 
    #    * 3 uur deleten...
    del_dir = "X:\\Monitoring\\OrthoSeg\\trees"
    
    time_start = dt.datetime.now()
    print(f"Start delete at {time_start}")
    shutil.rmtree(del_dir)
    time_taken = (dt.datetime.now()-time_start).total_seconds()/3600
    print(f"Time taken: {time_taken:.5f hours}")
    