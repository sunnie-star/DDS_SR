
import argparse

from PIL import Image, ImageDraw
from torchvision import io
import sys
from pathlib import Path
import yaml
import logging

sys.path.append('../')
from dds_utils import read_results_dict

relevant_classes = 'vehicle'
confidence_threshold = 0.5
max_area_threshold = 0.04
# iou_threshold = 0.8

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
    # results_file = 'results/trafficcam_2_dds_0.8_0.8_36_26_0.0_twosides_batch_15_0.5_0.3_0.01'
    # results_file = 'results/trafficcam_2_dds_0.8_0.8_36_26_0.0_twosides_batch_15_0.5_0.3_0.01_low_implement'
    results_file = 'images/trafficcam_1_dds_0.8_0.8_36_26_0.0_twosides_batch_15_0.5_0.3_0.01'
    results = read_results_dict(results_file)
    # folder to save image昂
    print(results_file)
    print(Path(results_file).name)  #  .name shi huode zhege wenjianjia d mingzi
    #  =====================================================================================================
    # save_folder = Path('visualize')/'final'
    # save_folder = Path('visualize') / 'low'
    # save_folder = Path('visualize') / 'high'
    # save_folder = Path('visualize')/'final_origion'
    # save_folder = Path('visualize') / 'low_origion'
    save_folder = Path('visualize') / 'resultimages'
    save_folder.mkdir(parents=True, exist_ok=True)
    
    for fid in range(max(results.keys())+1):  #  lus 1 is sun modified

        if fid % 10 == 0:
            logger.info(f'Visualizing image with frame id {fid}')
        
        # load origin image========================================================================================
        image_path=Path('images')/('%010d.png' % fid)
        image = Image.open(image_path)
        # drawer for this image
        draw = ImageDraw.Draw(image)

        width, height = image.size

        for region in results[fid]:
            x, y, w, h = region.x, region.y, region.w, region.h
            x1 = int(x * width + 0.5)
            x2 = int((x + w) * width + 0.5)
            y1 = int(y * height + 0.5)
            y2 = int((y + h) * height + 0.5)

            # filter out large regions, they are not true objects
            # if w * h > max_area_threshold:
            #     continue
            #
            # # filter out irrelevant regions
            # if region.label not in relevant_classes:
            #     continue
            #
            # # filter out low confidence regions
            # if region.conf < confidence_threshold:
            #     continue

            # default color
            color = '#318fb5'
            draw.rectangle([x1,y1,x2,y2], outline = color, width=10)
        
        image.save(save_folder / ('%010d.png' % fid))




if __name__ == '__main__':

    main()

    #  put origin images in 'image' folder

