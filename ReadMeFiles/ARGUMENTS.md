# Training - Command-line Flags

You can configure the behavior of the script by a variety of command-line flags:

## General Configurations
* `--N`: Specify number of particles. Default is 100.
* `--alpha`: Specify value fraction on how much we trust weighted estimate over initial target. Default is 0.0. 
* `--balance`: Specify whether or not balance filtering should be used. Default is `True`.
* `--thresh_c`: Specify estimation threshold for weights of color-based state vector. Default is 0.01.
* `--thresh_m`: Specify estimation threshold for weights of momentum-based state vector. Default is 0.01.
* `--log_info`: Specify whether to print information or not. Default is `False`.
* `--show_part`: Specify whether to show bounding boxes of all particles. Default is `False`.
* `--show_frames`: Specify whether to show every frame. Default is `False`. 
* `--show_GT`: Specify whether to show GT for every frame. Default is `True`.
* `--show_DT`: Specify whether to show result of pedestrian detection for every frame. Default is `False`.
* `--error`: Specify way of calculating the error between target and estimation. Choose either euclidean, overlap_area, area or both. Default is `overlap_area`.
* `--bestofboth`: Specify whether to pick best estimation results for both methods, using CM. Default is `False`.
* `--exp_nr`: Specify number of experiment. Default is 1.

## Configurations for detection
* `--detect`: Specify which detection method should be use. Default is `HOG`.
* `--colorbased`: Specify whether detection should include color information. Default is `True`.
* `--color`: Specify color for detection based on color information. Default is `BLACK`.

## Configurations for motion model
* `--dt`: Specify time step size. Default is 0.001. 

## Configurations for observation model
* `--OM`: Specify which observation model to use. Choose either CLR, MMT or CM. Default is `CLR`.
* `--mu_c`: Specify mean of color-based observation model. Default is 0.0.
* `--sigma_c`: Specify variance of color-based observation model. Default is 0.5.
* `--mu_m`: Specify mean of momentum-based observation model. Default is 0.0.
* `--sigma_m`: Specify variance of momentum-based observation model. Default is 0.5.
* `--beta`: Specify value for weight importance of fusion process. Default is 0.5.

## Configurations for resampling
* `--resampling`: Specify which resampling method to use. Default is `SYS`.

## Configurations for video generation
* `--video`:Specify whether to create video from frames or not. Default is `True`. 
* `--vid_name`: Specify name of video. Default is `""`. 
* `--videoloss`: Specify whether to show error in video. Default is `False`. 