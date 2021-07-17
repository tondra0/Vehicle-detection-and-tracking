Comp- 4301 Project [Meheroon Tondra 	201555661]
Source code and other materials
--------------------------------------------------------------------
final_report.pdf = final report

The project folder contains the following including this README file:

vehicle_detection.py = main python scripts
stratifiedcv.py = main python scripts
necessary_functions.py = this is needed to run vehicle_detection.py and stratifiedcv.py
necessary_functions_2.py = this is needed to run vehicle_detection.py 
parameters.py = class file for parameters, also needed to run vehicle_detection.py and stratifiedcv.py
The vehicles folder = vehicles images downloaded 
The non-vehicles folder = non-vehicles images downloaded
The vid_images folder = downloaded images extracted from _video.mp4
_video.mp4 = the video

The output folder:
This contains sll output images and the processed video produced after running the vehicle_detection.py and stratifiedcv.py.

The shell_src folder:
This contains two shell scripts (vehicle_job.sh and cv_job.sh) used to run vehicle_detection.py and stratifiedcv.py as Slurm jobs in compute canada. 
The account needs to be changed before running.
