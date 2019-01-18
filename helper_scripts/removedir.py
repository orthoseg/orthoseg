# -*- coding: utf-8 -*-
"""
Created on Sat Jan 12 13:05:32 2019

@author: pierog
"""

import os
import shutil
import datetime as dt

# If the script is ran directly...
if __name__ == '__main__':
    
    segment_subject = "horsetracks"
    base_dir = "X:\\PerPersoon\\PIEROG\\Taken\\2018\\2018-08-12_AutoSegmentation"
    project_dir = os.path.join(base_dir, segment_subject)
    training_dir = os.path.join(project_dir, "training")
    traindata_dir = os.path.join(training_dir, "train_14")
    predict_subdir = "horsetracks_14_inceptionresnetv2+linknet_0.91067_0.96048_0_eval_old6"
    predict_dir = os.path.join(traindata_dir, predict_subdir)
    
    time_start = dt.datetime.now()
    print(f"Start delete at {time_start}")
    shutil.rmtree(predict_dir)
    time_taken = (dt.datetime.now()-time_start).total_seconds()/1000
    print(f"Time taken: {time_taken:.5f}")
    