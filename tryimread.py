import cv2 as cv
import time
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
cap = cv.VideoCapture('t1.mp4')
width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv.CAP_PROP_FPS))
fourcc = cv.VideoWriter_fourcc(*'XVID')
print('width:', width, 'hight:', height, 'fps:', fps, 'fourcc:', fourcc)
writer = cv.VideoWriter('tresult.avi', fourcc, fps, (width, height))
t1=time.time()
cnt=0
while 1:
    ret,frame=cap.read()
    if not ret:
        break
    # cv.imwrite(f'{cnt}.png',frame)
    writer.write(frame)
    cnt=cnt+1

t2=time.time()  # 处理时间 8.17s   120帧，写一张图片几十毫秒;            video  1.57s   一帧十几毫秒
print(t2-t1)