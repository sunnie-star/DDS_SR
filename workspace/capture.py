import cv2 as cv
import os
import time
from datetime import datetime
# data_dir="../dataset"
# path= os.path.join(data_dir, 'my', 'src')
# video_name=os.path.join(path,"my.mp4")
video_name='t1.mp4'
cap=cv.VideoCapture(video_name)
# get camera or tmp
"""
    CV_CAP_PROP_POS_MSEC Current position of the video file in milliseconds.
    CV_CAP_PROP_POS_FRAMES 0-based index of the frame to be decoded/captured next.
    CV_CAP_PROP_POS_AVI_RATIO Relative position of the video file: 0 - start of the film, 1 - end of the film.
    CV_CAP_PROP_FRAME_WIDTH Width of the frames in the video stream.
    CV_CAP_PROP_FRAME_HEIGHT Height of the frames in the video stream.
    CV_CAP_PROP_FPS Frame rate.
    CV_CAP_PROP_FOURCC 4-character code of codec.
    CV_CAP_PROP_FRAME_COUNT Number of frames in the video file.
    CV_CAP_PROP_FORMAT Format of the Mat objects returned by retrieve() .
    CV_CAP_PROP_MODE Backend-specific value indicating the current capture mode.
    CV_CAP_PROP_BRIGHTNESS Brightness of the image (only for cameras).
    CV_CAP_PROP_CONTRAST Contrast of the image (only for cameras).
    CV_CAP_PROP_SATURATION Saturation of the image (only for cameras).
    CV_CAP_PROP_HUE Hue of the image (only for cameras).
    CV_CAP_PROP_GAIN Gain of the image (only for cameras).
    CV_CAP_PROP_EXPOSURE Exposure (only for cameras).
    CV_CAP_PROP_CONVERT_RGB Boolean flags indicating whether images should be converted to RGB.
    CV_CAP_PROP_WHITE_BALANCE Currently unsupported
    CV_CAP_PROP_RECTIFICATION Rectification flag for stereo cameras (note: only supported by DC1394 v 2.x backend cur- rently)
    """
cnt=0
t1=datetime.now()
time.sleep(2)
t2=datetime.now()
print(t2-t1)
width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv.CAP_PROP_FPS))
# fourcc=int(cap.get(cv.CAP_PROP_FOURCC))
# fourcc=cv.VideoWriter_fourcc(*"mp4v")   # 很小
fourcc=cv.VideoWriter_fourcc(*'x264')
# fourcc=cv.VideoWriter_fourcc('a', 'v', 'c', '1')
writer=cv.VideoWriter('video_resultlkkkkkkkkkkk.mp4',fourcc
                      ,fps,(width,height))


print('fourcc',fourcc,'fps',fps,'width',width,'height',height)
while 1:
    ret,frame=cap.read()
    if not ret:
        break
    # writer.write(frame)
    cv.imwrite('t.png',frame)

    # image_path=os.path.join(path,f"{str(cnt).zfill(10)}.png")
    # print(cnt, ':', datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'))
    # cv.imwrite(image_path,frame)
    # print(cnt, ':', datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'))
    cnt=cnt+1
    if cnt==15:
        break

cap.release()
# cv.destroyAllWindows()

#
# import cv2 as cv
#
# import numpy as np
#
# vid = cv.VideoWriter('pyout1.mp4', cv.VideoWriter_fourcc(*'avc1'), 25, (300, 300))
#
# # vid = cv.VideoWriter('pyout.mp4', 0x00000021, 25, (300,300))
#
#
# for i in range(250):
#     img = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
#
#     vid.write(img)
#
# vid.release()
#
