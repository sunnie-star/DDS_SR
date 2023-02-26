# import logging
# import os
# import shutil
# import requests
# import json
# import cv2 as cv
# import time
# from datetime import datetime
# from dds_utils import (Results, read_results_dict, cleanup, Region,
#                        compute_regions_size, extract_images_from_video,
#                        merge_boxes_in_results)
# regions=Results()
# regions.append(Region(
#     10, 0, 0, 1, 1, 1.0, 2,resolution=1))
# regions.append(Region(
#     9, 0, 0, 1, 1, 1.0, 2,resolution=1))
# print(regions)
# print(len(regions))
import shutil
# configuration.ml \ this file's trafficcam_1 every 9, overall 5
# t1
import os,sys
#
#old

otrafficname=3
ovideocnt=9
shutil.move('/home/sun/workplace/dds-clean/workspace/client.txt',
            f'/home/sun/桌面/sundesktop/视频集/trafficcam_{otrafficname}/client{ovideocnt}.txt')
shutil.move('/home/sun/workplace/dds-clean/workspace/results',
            f'/home/sun/桌面/sundesktop/视频集/trafficcam_{otrafficname}/results{ovideocnt}')

#new
# trafficname=3
# videocnt=9
# path=f'/home/sun/workplace/dds-clean/dataset/trafficcam_{trafficname}/src'
# shutil.rmtree(path)
# os.makedirs(path, exist_ok=True)
# shutil.copy(f'/home/sun/桌面/sundesktop/视频集/trafficcam_{trafficname}/t{videocnt}.mp4',
#             path+f'/trafficcam_{trafficname}.mp4')
