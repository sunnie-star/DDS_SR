
import argparse

from PIL import Image, ImageDraw
from torchvision import io
import sys
from pathlib import Path
import yaml
import logging
import cv2 as cv
import numpy as np

sys.path.append('../')
from dds_utils import read_results_dict

relevant_classes = 'vehicle'
confidence_threshold = 0.5
max_area_threshold = 0.04
iou_threshold = 0.8

def iou(b1, b2):
    # calculate the iou of two bounding boxes
	(x1,y1,w1,h1) = b1
	(x2,y2,w2,h2) = b2
	x3 = max(x1,x2)
	y3 = max(y1,y2)
	x4 = min(x1+w1,x2+w2)
	y4 = min(y1+h1,y2+h2)
	if x3>x4 or y3>y4:
		return 0
	else:
		overlap = (x4-x3)*(y4-y3)
		return overlap/(w1*h1+w2*h2-overlap)

def main():

    # get logger
    logging.basicConfig(
        format="%(name)s -- %(levelname)s -- %(lineno)s -- %(message)s",
        level='INFO')

    logger = logging.getLogger("visualize")
    logger.addHandler(logging.NullHandler())

    # load configuration file to get the directory of dataset
    with open('configuration.yml', 'r') as f:
        config = yaml.load(f.read())

    # load regions from result_file=========================================================================
    results_file = 'video_origin/all_image_SR_file'
    # results_file = 'results/trafficcam_2_dds_0.8_0.8_36_26_0.0_twosides_batch_15_0.5_0.3_0.01_low_implement'
    # results_file = 'results/trafficcam_2_dds_0.8_0.8_36_26_0.0_twosides_batch_15_0.5_0.3_0.01_high_implement'
    results = read_results_dict(results_file)
    # folder to save image昂
    print(results_file)
    print(Path(results_file).name)  #  .name shi huode zhege wenjianjia d mingzi
    #  =====================================================================================================
    video_path='video_origin/trafficcam_1.mp4'
    cap = cv.VideoCapture(video_path)
    width=int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height=int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    fps=int(cap.get(cv.CAP_PROP_FPS))

    # fourcc=cv.VideoWriter_fourcc('X','2','6','4')
    # 这个选项是一种比较新的MPEG-4编码方式。如果你想限制结果视频的大小，这可能是最好的选择。文件扩展名应为.mp4。

    # fourcc=cv.VideoWriter_fourcc('I', '4', '2', '0')
    #     这个选项是一个未压缩的YUV编码，4: 2:0
    #     色度子采样。这种编码广泛兼容，但会产生大文件。文件扩展名应为.avi。

    # # cv2.VideoWriter_fourcc('P','I','M','1')
    # 此选项为MPEG-1。文件扩展名应为.avi

    fourcc=cv.VideoWriter_fourcc('X', 'V', 'I', 'D')
    # 此选项是一个相对较旧的MPEG - 4
    # 编码。如果要限制结果视频的大小，这是一个很好的选择。文件扩展名应为.avi。

    # fourcc = cv.VideoWriter_fourcc('M', 'P', '4', 'V')
    # 此选项是另一个相对较旧的MPEG-4编码。如果要限制结果视频的大小，这是一个很好的选择。文件扩展名应为.mp4。

    # print('width:',width,'hight:',height,'fps:',fps,'fourcc:',fourcc)
    writer=cv.VideoWriter('video_visual/video_result_all_image_SR_file.avi',fourcc,fps,(width,height))
    # writer = cv.VideoWriter('video_visual/video_result.mp4', fourcc, fps, (width, height))
    fid=-1
    while 1:
        ret,frame=cap.read()   #  opencvall BGR
        if ret:
            fid=fid+1
            image= Image.fromarray(cv.cvtColor(frame, cv.COLOR_BGR2RGB))  # PIL  all RGB,so convert color mode
            # print('image.mode',image.mode)
        else:
            print(fid)
            break
        #opencv读取图片的颜色空间和数据格式不清楚，他们里面有BGR和RGB颜色空间，而一般cv2.imread()读取的图片都是BGR颜色空间的图片，cv2.VideoCapture()获取的视频帧是RGB颜色空间的图片。PIL（Python Image Library）读取的图片是RGB颜色空间的。
        #opencv读取的图片不管是视频帧还是图片都是矩阵形式，即np.array，转PIL.Image格式用PIL.Image.fromarray()函数即可。
        if fid % 10 == 0:
            logger.info(f'Visualizing image with frame id {fid}')
            # drawer for this image
        draw = ImageDraw.Draw(image)
        width, height = image.size
        # for region in results[fid]:
        #     x, y, w, h = region.x, region.y, region.w, region.h
        #     x1 = int(x * width + 0.5)
        #     x2 = int((x + w) * width + 0.5)
        #     y1 = int(y * height + 0.5)
        #     y2 = int((y + h) * height + 0.5)
        #
        #     # filter out large regions, they are not true objects
        #     if w * h > max_area_threshold:
        #         continue
        #
        #     # filter out irrelevant regions
        #     if region.label not in relevant_classes:
        #         continue
        #
        #     # filter out low confidence regions
        #     if region.conf < confidence_threshold:
        #         continue
        #
        #     # default color
        #     color = '#318fb5'
        #     draw.rectangle([x1, y1, x2, y2], outline=color,width=10)
        #
        # f=cv.cvtColor(np.array(image), cv.COLOR_RGB2BGR)
        # # image.save('tmp.jpg')
        # # f=cv.imread('tmp.jpg')
        # writer.write(f)


        if fid in results.keys():
            for region in results[fid]:
                x, y, w, h = region.x, region.y, region.w, region.h
                x1 = int(x * width + 0.5)
                x2 = int((x + w) * width + 0.5)
                y1 = int(y * height + 0.5)
                y2 = int((y + h) * height + 0.5)

                # filter out large regions, they are not true objects
                if w * h > max_area_threshold:
                    continue

                # filter out irrelevant regions
                if region.label not in relevant_classes:
                    continue

                # filter out low confidence regions
                if region.conf < confidence_threshold:
                    continue

                # default color
                color = '#318fb5'
                draw.rectangle([x1, y1, x2, y2], outline=color,width=10)

            f=cv.cvtColor(np.array(image), cv.COLOR_RGB2BGR)
            # image.save('tmp.jpg')
            # f=cv.imread('tmp.jpg')
            writer.write(f)
        else:
            writer.write(frame)


    
    # for fid in range(max(results.keys())+1):  #  lus 1 is sun modified
    #
    #     if fid % 10 == 0:
    #         logger.info(f'Visualizing image with frame id {fid}')
    #
    #     # load origin mage========================================================================================
    #     image_path=Path('images')/('%010d.png' % fid)
    #     image = Image.open(image_path)
    #     # drawer for this image
    #     draw = ImageDraw.Draw(image)
    #
    #     width, height = image.size
    #
    #     for region in results[fid]:
    #         x, y, w, h = region.x, region.y, region.w, region.h
    #         x1 = int(x * width + 0.5)
    #         x2 = int((x + w) * width + 0.5)
    #         y1 = int(y * height + 0.5)
    #         y2 = int((y + h) * height + 0.5)
    #
    #         # filter out large regions, they are not true objects
    #         if w * h > max_area_threshold:
    #             continue
    #
    #         # filter out irrelevant regions
    #         if region.label not in relevant_classes:
    #             continue
    #
    #         # filter out low confidence regions
    #         if region.conf < confidence_threshold:
    #             continue
    #
    #         # default color
    #         color = '#318fb5'
    #         draw.rectangle([x1,y1,x2,y2], outline = color, width=10)
    #
    #     image.save(save_folder / ('%010d.png' % fid))




if __name__ == '__main__':

    main()

    #  put origin video in 'video_origin' folder

