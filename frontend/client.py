import logging
import os
import shutil
import requests
import json
import cv2 as cv
import time
from datetime import datetime
from dds_utils import (Results, read_results_dict, cleanup, Region,
                       compute_regions_size, extract_images_from_video,
                       merge_boxes_in_results,evaluate)
import yaml


class Client:
    """The client of the DDS protocol
       sends images in low resolution and waits for
       further instructions from the server. And finally receives results
       Note: All frame ranges are half open ranges"""

    def __init__(self, hname, config, server_handle=None):
        if hname:
            self.hname = hname
            self.session = requests.Session()
        else:
            self.server = server_handle
        self.config = config

        self.logger = logging.getLogger("client")
        handler = logging.NullHandler()
        self.logger.addHandler(handler)

        self.logger.info(f"Client initialized")

    def analyze_video_mpeg(self, video_name, raw_images_path, enforce_iframes):
        print("raw_images_path", raw_images_path)  #../dataset/my/src
        video_path = os.path.join(raw_images_path, self.config.video_oname+".mp4")
        print("video_path", video_path)
        cap=cv.VideoCapture(video_path)
        number_of_frames=int(cap.get(7))
        fps=int(cap.get(5))
        print("number_of_frames",number_of_frames)
        print("fps", fps)
        # origin
        final_results = Results()
        final_rpn_results = Results()
        total_size = 0
        # origin
        cnt = 0
        while 1:
            if cnt>=number_of_frames:
                break
            ret, frame = cap.read()
            image_path = os.path.join(raw_images_path, f"{str(cnt).zfill(10)}.png")
            if ret:
                cv.imwrite(image_path, frame)
            else:
                print("cannot open image")
            cnt=cnt+1    # cnt represent saved count  eg.15  0-14
            if (cnt % self.config.batch_size == 0) or (cnt == number_of_frames):
                end_frame = cnt
                if cnt%self.config.batch_size==0:
                  start_frame = cnt-self.config.batch_size
                else:
                  start_frame = cnt-cnt%self.config.batch_size
                self.logger.info(f"Processing batch from {start_frame} to {end_frame - 1}")
                batch_fnames = sorted([f"{str(idx).zfill(10)}.png"
                                       for idx in range(start_frame, end_frame)])

                req_regions = Results()
                for fid in range(start_frame, end_frame):
                    req_regions.append(
                        Region(fid, 0, 0, 1, 1, 1.0, 2,
                               self.config.low_resolution))
                batch_video_size, _ = compute_regions_size(
                    req_regions, f"{video_name}-base-phase", raw_images_path,
                    self.config.low_resolution, self.config.low_qp,
                    enforce_iframes, True)
                self.logger.info(f"{batch_video_size / 1024}KB sent "
                                 f"in base phase using {self.config.low_qp}QP")
                extract_images_from_video(f"{video_name}-base-phase-cropped",
                                          req_regions)
                results, rpn_results = (
                    self.server.perform_detection(
                        f"{video_name}-base-phase-cropped",
                        self.config.low_resolution, batch_fnames))

                self.logger.info(f"Detection {len(results)} regions for "
                                 f"batch {start_frame} to {end_frame-1} with a "
                                 f"total size of {batch_video_size / 1024}KB")

                # fo=open('gtresults.txt','a+')
                # fo.write(f'{len(results)}\n')
                # fo.close()

                final_results.combine_results(
                    results, self.config.intersection_threshold)
                final_rpn_results.combine_results(
                    rpn_results, self.config.intersection_threshold)

                # Remove encoded video manually
                shutil.rmtree(f"{video_name}-base-phase-cropped")
                total_size += batch_video_size

        # up is my
        # number_of_frames = len(
        #     [f for f in os.listdir(raw_images_path) if ".png" in f])

        # for i in range(0, number_of_frames, self.config.batch_size):
        #     start_frame = i
        #     end_frame = min(number_of_frames, i + self.config.batch_size)
        #
        #     batch_fnames = sorted([f"{str(idx).zfill(10)}.png"
        #                            for idx in range(start_frame, end_frame)])
        #
        #     req_regions = Results()
        #     for fid in range(start_frame, end_frame):
        #         req_regions.append(
        #             Region(fid, 0, 0, 1, 1, 1.0, 2,
        #                    self.config.low_resolution))
        #     batch_video_size, _ = compute_regions_size(
        #         req_regions, f"{video_name}-base-phase", raw_images_path,
        #         self.config.low_resolution, self.config.low_qp,
        #         enforce_iframes, True)
        #     self.logger.info(f"{batch_video_size / 1024}KB sent "
        #                      f"in base phase using {self.config.low_qp}QP")
        #     extract_images_from_video(f"{video_name}-base-phase-cropped",
        #                               req_regions)
        #     results, rpn_results = (
        #         self.server.perform_detection(
        #             f"{video_name}-base-phase-cropped",
        #             self.config.low_resolution, batch_fnames))
        #
        #     self.logger.info(f"Detection {len(results)} regions for "
        #                      f"batch {start_frame} to {end_frame} with a "
        #                      f"total size of {batch_video_size / 1024}KB")
        #     final_results.combine_results(
        #         results, self.config.intersection_threshold)
        #     final_rpn_results.combine_results(
        #         rpn_results, self.config.intersection_threshold)
        #
        #     # Remove encoded video manually
        #     shutil.rmtree(f"{video_name}-base-phase-cropped")
        #     total_size += batch_video_size

        final_results = merge_boxes_in_results(
            final_results.regions_dict, 0.3, 0.3)
        final_results.fill_gaps(number_of_frames)

        # Add RPN regions
        final_results.combine_results(
            final_rpn_results, self.config.intersection_threshold)

        final_results.write(video_name)

        return final_results, [total_size, 0]

    def analyze_video_emulate(self, video_name, high_images_path,
                              enforce_iframes, low_results_path=None,
                              debug_mode=False):
        print("video_name",video_name,'enforce_iframes',enforce_iframes,
              'low_results_path',low_results_path,'debug_mode',debug_mode)
        print("raw_images_path", high_images_path)  # ../dataset/my/src
        video_path = os.path.join(high_images_path, self.config.video_oname+".mp4")
        print("video_path", video_path)
        cap = cv.VideoCapture(video_path)
        number_of_frames = int(cap.get(7))
        fps = int(cap.get(5))
        print("number_of_frames", number_of_frames)
        print("fps", fps)
        final_results = Results()
        low_phase_results = Results()
        high_phase_results = Results()

        # number_of_frames = len(
        #     [x for x in os.listdir(high_images_path) if "png" in x])


        low_results_dict = None
        if low_results_path:
            low_results_dict = read_results_dict(low_results_path)

        total_size = [0, 0]
        total_regions_count = 0
        cnt=0
        while 1:
            if cnt>=number_of_frames:
                break
            ret,frame=cap.read()
            image_path = os.path.join(high_images_path, f"{str(cnt).zfill(10)}.png")
            if ret:
                cv.imwrite(image_path, frame)
            else:
                print("cannot open image")
            cnt=cnt+1    # cnt represent saved count  eg.15  0-14
            if (cnt % self.config.batch_size == 0) or (cnt == number_of_frames):
                end_fid = cnt
                if cnt%self.config.batch_size==0:
                  start_fid = cnt-self.config.batch_size
                else:
                  start_fid = cnt-cnt%self.config.batch_size
                self.logger.info(f"Processing batch from {start_fid} to {end_fid-1}")
                # Encode frames in batch and get size
                # Make temporary frames to downsize complete frames
                base_req_regions = Results()
                for fid in range(start_fid, end_fid):
                    base_req_regions.append(
                        Region(fid, 0, 0, 1, 1, 1.0, 2,
                               self.config.high_resolution))
                encoded_batch_video_size, batch_pixel_size = compute_regions_size(
                    base_req_regions, f"{video_name}-base-phase", high_images_path,
                    self.config.low_resolution, self.config.low_qp,
                    enforce_iframes, True)
                self.logger.info(f"Sent {encoded_batch_video_size / 1024} "
                                 f"in base phase")
                total_size[0] += encoded_batch_video_size

                # Low resolution phase
                low_images_path = f"{video_name}-base-phase-cropped"
                r1, req_regions = self.server.simulate_low_query(
                    start_fid, end_fid, low_images_path, low_results_dict, False,
                    self.config.rpn_enlarge_ratio)
                total_regions_count += len(req_regions)
                low_phase_results.combine_results(
                    r1, self.config.intersection_threshold)
                final_results.combine_results(
                    r1, self.config.intersection_threshold)

                # High resolution phase
                if len(req_regions) > 0:
                    # Crop, compress and get size
                    regions_size, _ = compute_regions_size(
                        req_regions, video_name, high_images_path,
                        self.config.high_resolution, self.config.high_qp,
                        enforce_iframes, True)
                    self.logger.info(f"Sent {len(req_regions)} regions which have "
                                     f"{regions_size / 1024}KB in second phase "
                                     f"using {self.config.high_qp}")
                    total_size[1] += regions_size

                    # High resolution phase every three filter
                    r2 = self.server.emulate_high_query(
                        video_name, low_images_path, req_regions)
                    self.logger.info(f"Got {len(r2)} results in second phase "
                                     f"of batch")
                    high_phase_results.combine_results(
                        r2, self.config.intersection_threshold)
                    final_results.combine_results(
                        r2, self.config.intersection_threshold)

                # Cleanup for the next batch
                cleanup(video_name, debug_mode, start_fid, end_fid)

        # for i in range(0, number_of_frames, self.config.batch_size):
        #     start_fid = i
        #     end_fid = min(number_of_frames, i + self.config.batch_size)
        #     self.logger.info(f"Processing batch from {start_fid} to {end_fid}")
        #
        #     # Encode frames in batch and get size
        #     # Make temporary frames to downsize complete frames
        #     base_req_regions = Results()
        #     for fid in range(start_fid, end_fid):
        #         base_req_regions.append(
        #             Region(fid, 0, 0, 1, 1, 1.0, 2,
        #                    self.config.high_resolution))
        #     encoded_batch_video_size, batch_pixel_size = compute_regions_size(
        #         base_req_regions, f"{video_name}-base-phase", high_images_path,
        #         self.config.low_resolution, self.config.low_qp,
        #         enforce_iframes, True)
        #     self.logger.info(f"Sent {encoded_batch_video_size / 1024} "
        #                      f"in base phase")
        #     total_size[0] += encoded_batch_video_size
        #
        #     # Low resolution phase
        #     low_images_path = f"{video_name}-base-phase-cropped"
        #     r1, req_regions = self.server.simulate_low_query(
        #         start_fid, end_fid, low_images_path, low_results_dict, False,
        #         self.config.rpn_enlarge_ratio)
        #     total_regions_count += len(req_regions)
        #
        #     low_phase_results.combine_results(
        #         r1, self.config.intersection_threshold)
        #     final_results.combine_results(
        #         r1, self.config.intersection_threshold)
        #
        #     # High resolution phase
        #     if len(req_regions) > 0:
        #         # Crop, compress and get size
        #         regions_size, _ = compute_regions_size(
        #             req_regions, video_name, high_images_path,
        #             self.config.high_resolution, self.config.high_qp,
        #             enforce_iframes, True)
        #         self.logger.info(f"Sent {len(req_regions)} regions which have "
        #                          f"{regions_size / 1024}KB in second phase "
        #                          f"using {self.config.high_qp}")
        #         total_size[1] += regions_size
        #
        #         # High resolution phase every three filter
        #         r2 = self.server.emulate_high_query(
        #             video_name, low_images_path, req_regions)
        #         self.logger.info(f"Got {len(r2)} results in second phase "
        #                          f"of batch")
        #
        #         high_phase_results.combine_results(
        #             r2, self.config.intersection_threshold)
        #         final_results.combine_results(
        #             r2, self.config.intersection_threshold)
        #
        #     # Cleanup for the next batch
        #     cleanup(video_name, debug_mode, start_fid, end_fid)

        self.logger.info(f"Got {len(low_phase_results)} unique results "
                         f"in base phase")
        self.logger.info(f"Got {len(high_phase_results)} positive "
                         f"identifications out of {total_regions_count} "
                         f"requests in second phase")

        # Fill gaps in results
        final_results.fill_gaps(number_of_frames)

        # Write results
        final_results.write(f"{video_name}")

        self.logger.info(f"Writing results for {video_name}")
        self.logger.info(f"{len(final_results)} objects detected "
                         f"and {total_size[1]} total size "
                         f"of regions sent in high resolution")

        rdict = read_results_dict(f"{video_name}")
        final_results = merge_boxes_in_results(rdict, 0.3, 0.3)

        final_results.fill_gaps(number_of_frames)
        final_results.write(f"{video_name}")
        low_phase_results.write(f"{video_name}_low")
        high_phase_results.write(f"{video_name}_high")
        return final_results, total_size

    def init_server(self, nframes):
        self.config['nframes'] = nframes
        response = self.session.post(
            "http://" + self.hname + "/init", data=yaml.dump(self.config))
        if response.status_code != 200:
            self.logger.fatal("Could not initialize server")
            # Need to add exception handling
            exit()

    """
    def get_first_phase_results(self, vid_name):
        # nd temp.mp4
        encoded_vid_path = os.path.join(
            vid_name + "-base-phase-cropped", "temp.mp4")
        t1 = datetime.now()
        print('1 begin open send data-------------------------------------', t1)
        video_to_send = {"media": open(encoded_vid_path, "rb")}
        t1 = datetime.now()
        print('1 begin send data------------------------------------------', t1)
        #
        # fo=open('client.txt','a+')
        # fo.write(f'1 begin send data:{t1}\n')
        # fo.close()  # need to close , because then write server.txt

        response = self.session.post(
            "http://" + self.hname + "/low", files=video_to_send)
        t1=datetime.now()
        print('1 end receive results--------------------------------------', t1)

        # fo = open('client.txt', 'a+')
        # fo.write(f'1 end receive results:{t1}\n')
        # fo.close()

        # load chu cuo ,keng json shi none  其次查看是否处理的数据是否是json,比如你使用接口返回的数据，返回的数据为空，而你却在使用json解析函数在处理
        response_json = json.loads(response.text)
        results = Results()
        for region in response_json["results"]:
            results.append(Region.convert_from_server_response(
                region, self.config.low_resolution, "low-res"))
        rpn = Results()
        for region in response_json["req_regions"]:
            rpn.append(Region.convert_from_server_response(
                region, self.config.low_resolution, "low-res"))

        return results, rpn
    """

    #  SR =====================================================
    def get_first_phase_results(self, vid_name):
        # nd temp.mp4
        encoded_vid_path = os.path.join(
            vid_name + "-base-phase-cropped", "temp.mp4")
        t1 = datetime.now()
        print('1 begin open send data-------------------------------------', t1)
        video_to_send = {"media": open(encoded_vid_path, "rb")}
        t1 = datetime.now()
        print('1 begin send data------------------------------------------', t1)
        #
        # fo=open('client.txt','a+')
        # fo.write(f'1 begin send data:{t1}\n')
        # fo.close()  # need to close , because then write server.txt

        response = self.session.post(
            "http://" + self.hname + "/low", files=video_to_send)
        t1 = datetime.now()
        print('1 end receive results--------------------------------------', t1)

        # fo = open('client.txt', 'a+')
        # fo.write(f'1 end receive results:{t1}\n')
        # fo.close()

        # load chu cuo ,keng json shi none  其次查看是否处理的数据是否是json,比如你使用接口返回的数据，返回的数据为空，而你却在使用json解析函数在处理
        response_json = json.loads(response.text)
        results = Results()
        for region in response_json["results"]:
            results.append(Region.convert_from_server_response(
                region, self.config.low_resolution, "low-res"))
        return results
    # SR===========================================================================================

    def get_second_phase_results(self, vid_name):
        encoded_vid_path = os.path.join(vid_name + "-cropped", "temp.mp4")
        print("get_second_phase_results encoded_vid_path",encoded_vid_path)
        t1 = datetime.now()
        print('2 begin open send data-------------------------------------', t1)
        video_to_send = {"media": open(encoded_vid_path, "rb")}
        t1 = datetime.now()
        print('2 begin send data------------------------------------------', t1)

        # fo = open('client.txt', 'a+')
        # fo.write(f'2 begin send data:{t1}\n')
        # fo.close()

        response = self.session.post(
            "http://" + self.hname + "/high", files=video_to_send)
        t1=datetime.now()
        print('2 end receive results--------------------------------------', t1)

        # fo = open('client.txt', 'a+')
        # fo.write(f'2 end receive results:{t1}\n')
        # fo.close()

        response_json = json.loads(response.text)
        results = Results()
        for region in response_json["results"]:
            results.append(Region.convert_from_server_response(
                region, self.config.high_resolution, "high-res"))

        return results

    def analyze_video(
            self, vid_name, raw_images, config, enforce_iframes):
        # fo = open('client.txt', 'a+')
        # fo.write(f'vid_name:{vid_name}\n')
        # fo.close()
        # results/trafficcam_1_dds_low res_high res_low qp_high_qp_rpn enlarge ratio_twosides_batch ze_
        # prune score_objfilter iou_size obj
        print("raw_images_path", raw_images)  # ../dataset/trafficcam_1/src
        video_path = os.path.join(raw_images, self.config.video_oname+".mp4")
        print("video_path", video_path)
        cap = cv.VideoCapture(video_path)
        nframes = int(cap.get(7))
        fps = int(cap.get(5))
        print("number_of_frames", nframes)
        print("fps", fps)
        final_results = Results()
        low_phase_results = Results()
        high_phase_results = Results()
        all_required_regions = Results()
        low_phase_size = 0
        high_phase_size = 0
        # nframes = sum(map(lambda e: "png" in e, os.listdir(raw_images)))
        self.init_server(nframes)
        total_regions_count=0
        cnt=0
        while 1:
            if cnt>=nframes:
                break
            ret,frame=cap.read()
            image_path = os.path.join(raw_images, f"{str(cnt).zfill(10)}.png")
            if ret:
                cv.imwrite(image_path, frame)
            else:
                print("cannot open image")
            cnt = cnt + 1  # cnt represent saved count  eg.15  0-14
            if (cnt % self.config.batch_size == 0) or (cnt == nframes):
                end_frame= cnt
                if cnt % self.config.batch_size == 0:
                    start_frame = cnt - self.config.batch_size
                else:
                    start_frame = cnt - cnt % self.config.batch_size
                print('=============================================================')
                self.logger.info(f"Processing frames {start_frame} to {end_frame-1}")
                # First iteration
                req_regions = Results()
                for fid in range(start_frame, end_frame):                    req_regions.append(Region(
                        fid, 0, 0, 1, 1, 1.0, 2, self.config.low_resolution))
                t1 = datetime.now()
                print('1 begin compute_regions_size -------------------------------------', t1)
                batch_video_size, _ = compute_regions_size(
                    req_regions, f"{vid_name}-base-phase", raw_images,
                    self.config.low_resolution, self.config.low_qp,
                    enforce_iframes, True)
                t2 = datetime.now()
                print('1 end compute_regions_size -------------------------------------',t2,'--',t2-t1)

                #  write file 1 encode time
                # fo = open('client.txt', 'a+')
                # fo.write(f'1 encode time:{t2-t1}\n')
                # fo.close()

                low_phase_size += batch_video_size
                self.logger.info(f"{batch_video_size / 1024}KB sent in base phase."
                                 f"Using QP {self.config.low_qp} and "
                                 f"Resolution {self.config.low_resolution}.")
                # SR ===================================================================================
                results= self.get_first_phase_results(vid_name)  # receive server's response.text
                final_results.combine_results(
                    results, self.config.intersection_threshold)
                # SR ===================================================================================

                """
                results, rpn_regions = self.get_first_phase_results(vid_name)  # receive server's response.text
                total_regions_count += len(rpn_regions)
                low_phase_results.combine_results(
                    results, self.config.intersection_threshold)
                final_results.combine_results(
                    results, self.config.intersection_threshold)
                all_required_regions.combine_results(
                    rpn_regions, self.config.intersection_threshold)

                self.logger.info(f'frome frame {start_frame} to {end_frame-1}  rpn_regions:{len(rpn_regions)}')

                # fo=open('rpn.txt','a+')
                # fo.write(f'{len(rpn_regions)}\n')
                # fo.close()

                self.logger.info(f'frome frame {start_frame} to {end_frame - 1}  low results:{len(results)}')
                # fo = open('lowresults.txt', 'a+')
                # fo.write(f'{len(results)}\n')
                # fo.close()
                # use SR, below disgard
                # Second Iteration
                if len(rpn_regions) > 0:
                    t1=datetime.now()
                    print('2 begin compute_regions_size -------------------------------------', t1)
                    batch_video_size, _ = compute_regions_size(
                        rpn_regions, vid_name, raw_images,
                        self.config.high_resolution, self.config.high_qp,
                        enforce_iframes, True)
                    t2 = datetime.now()
                    print('2 end compute_regions_size -------------------------------------', t2, '--', t2 - t1)
                    #  write file 1 encode time
                    # fo = open('client.txt', 'a+')
                    # fo.write(f'2 encode time:{t2 - t1}\n')
                    # fo.close()

                    high_phase_size += batch_video_size
                    self.logger.info(f"{batch_video_size / 1024}KB sent in second "
                                     f"phase. Using QP {self.config.high_qp} and "
                                     f"Resolution {self.config.high_resolution}.")
                    results = self.get_second_phase_results(vid_name)
                    high_phase_results.combine_results(
                        results, self.config.intersection_threshold)
                    final_results.combine_results(
                        results, self.config.intersection_threshold)
                    self.logger.info(
                        f'frome frame {start_frame} to {end_frame - 1} high results:{len(results)}')
                    
                    # fo = open('highresults.txt', 'a+')
                    # fo.write(f'{len(results)}\n')
                    # fo.close()

                # Cleanup for the next batch,woba debug mode gaicheng truel
                cleanup(vid_name,False , start_frame, end_frame)

        # for i in range(0, nframes, self.config.batch_size):
        #     start_frame = i
        #     end_frame = min(nframes, i + self.config.batch_size)
        #     self.logger.info(f"Processing frames {start_frame} to {end_frame}")
        #
        #     # First iteration
        #     req_regions = Results()
        #     for fid in range(start_frame, end_frame):
        #         req_regions.append(Region(
        #             fid, 0, 0, 1, 1, 1.0, 2, self.config.low_resolution))
        #     batch_video_size, _ = compute_regions_size(
        #         req_regions, f"{vid_name}-base-phase", raw_images,
        #         self.config.low_resolution, self.config.low_qp,
        #         enforce_iframes, True)
        #     low_phase_size += batch_video_size
        #     self.logger.info(f"{batch_video_size / 1024}KB sent in base phase."
        #                      f"Using QP {self.config.low_qp} and "
        #                      f"Resolution {self.config.low_resolution}.")
        #     results, rpn_regions = self.get_first_phase_results(vid_name)
        #     final_results.combine_results(
        #         results, self.config.intersection_threshold)
        #     all_required_regions.combine_results(
        #         rpn_regions, self.config.intersection_threshold)
        #
        #     # Second Iteration
        #     if len(rpn_regions) > 0:
        #         batch_video_size, _ = compute_regions_size(
        #             rpn_regions, vid_name, raw_images,
        #             self.config.high_resolution, self.config.high_qp,
        #             enforce_iframes, True)
        #         high_phase_size += batch_video_size
        #         self.logger.info(f"{batch_video_size / 1024}KB sent in second "
        #                          f"phase. Using QP {self.config.high_qp} and "
        #                          f"Resolution {self.config.high_resolution}.")
        #         results = self.get_second_phase_results(vid_name)
        #         final_results.combine_results(
        #             results, self.config.intersection_threshold)
        #
        #     # Cleanup for the next batch
        #     cleanup(vid_name, False, start_frame, end_frame)
        # yizhengge video de
        self.logger.info(f"Got {len(low_phase_results)} unique results "
                         f"in base phase")
        self.logger.info(f"Got {len(high_phase_results)} positive "
                         f"identifications out of {total_regions_count} "
                         f"requests in second phase")
        # total_regions_count += len(rpn_regions)  RPN 总数

        self.logger.info(f"Merging results")
        final_results = merge_boxes_in_results(
            final_results.regions_dict, 0.3, 0.3)
        self.logger.info(f"Writing results for {vid_name}")
        final_results.fill_gaps(nframes)

        final_results.combine_results(
            all_required_regions, self.config.intersection_threshold)
        final_results.write(f"{vid_name}")
        low_phase_results.write(f"{vid_name}_low_implement")
        high_phase_results.write(f"{vid_name}_high_implement")
        return final_results, low_phase_results,(low_phase_size, high_phase_size)
        """
        final_results.write(f"{vid_name}")
        return final_results,low_phase_size
