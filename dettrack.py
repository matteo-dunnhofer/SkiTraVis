import time
import numpy as np
from PIL import Image

from utils.general import scale_boxes, xywh2xyxy_tl, xyxy2xywh_tl

class DetTrack(object):

    def __init__(self, detector, tracker, verbose=False):
        self.detector = detector
        self.tracker = tracker

        self.id_counter = 0

        self.found = False

        self.tracking_conf_thresh = 0.5
        self.last_n_scores = 10
        self.track_fail_maximum = 30
        self.save_last_box_every = 5

        self.verbose = verbose

        self.t = 0

    def warmup(self, imgsz):
        self.detector.warmup(imgsz)

    def update(self, det_frame=None, track_frame=None, detection_area_box=None, init_box=None):
        """
        Receives a frame and returns:
        - a boolean whether a target skier is found
        - the ID of the skier
        - the bounding-box of the skier
        """
        self.t += 1

        if not self.found:

            if init_box is None:
                st_time = time.time()
                box, score = self.detector.detect(det_frame)
                det_time = time.time() - st_time

                if self.verbose:
                    print(f'FRAME #{self.t-1} - DETECTOR run in {np.round(det_time, 3)}s')
            else:
                box = init_box
                score = 1.0

            if (detection_area_box is not None) and (box is not None):
                c_x = box[0] + (box[2] / 2)
                c_y = box[1] + (box[3] / 2)

                if (detection_area_box[0] <= c_x <= detection_area_box[2]) and (detection_area_box[1] <= c_y <= detection_area_box[3]):
                    pass
                else:
                    # found detection out of the given area
                    box, score = None, None

            if box is None:
                return False, None, None, None
            else:
                # target found
                self.found = True
                
                if sum(det_frame.shape) != sum(track_frame.shape):
                    box = xywh2xyxy_tl(np.expand_dims(box, 0))
                    box = scale_boxes(det_frame.shape[:2], box, track_frame.shape).round()
                    box = xyxy2xywh_tl(box)[0]
                
                st_time = time.time()
                # init the tracker
                self.tracker.init(track_frame, box)
                tr_init_time = time.time() - st_time

                self.tracker_scores = []
                self.track_fail_counter = 0

                self.last_box = box

                if self.verbose:
                    print(f'FRAME #{self.t-1} - TARGET ID {self.id_counter} detected at {str(box)} with score {score}')
                    print(f'FRAME #{self.t-1} - TRACKER initialized in {np.round(tr_init_time, 3)}s')

                # return target localization and info
                return True, self.id_counter, np.copy(box), score
        else:
            st_time = time.time()
            box, score = self.tracker.update(track_frame)
            tr_time = time.time() - st_time

            self.tracker_scores.append(score)
            if len(self.tracker_scores) > self.last_n_scores:
                tr_score = np.mean(self.tracker_scores[-self.last_n_scores:])
            else:
                tr_score = score
            
            if self.verbose:
                print(f'FRAME #{self.t-1} - TRACKER run in {np.round(tr_time, 3)}s')

            pts = np.array([[self.last_box[0] + self.last_box[2] / 2, self.last_box[1] + self.last_box[3] / 2], 
                                            [box[0] + box[2] / 2, box[1] + box[3] / 2]])
            next_coord = self.next_point(pts)

            if next_coord[0] <= 0 or next_coord[1] <= 0 or next_coord[0] >= track_frame.shape[1] or next_coord[1] >= track_frame.shape[0]:

                if self.verbose:
                    print(f'FRAME #{self.t-1} - **** TARGET {self.id_counter} out of the frame')

                self.id_counter += 1
                self.found = False
                
                return False, None, None, None
            elif tr_score >= self.tracking_conf_thresh:

                self.track_fail_counter = 0

                if self.t % self.save_last_box_every == 0:
                    self.last_box = box

                # valid tracking
                if self.verbose:
                    print(f'FRAME #{self.t-1} - TARGET ID {self.id_counter} tracked at {str(box)} with score {tr_score}')

                return True, self.id_counter, np.copy(box), tr_score
            elif self.track_fail_counter > self.track_fail_maximum:
                # check for stopping criteria

                if self.verbose:
                    print(f'FRAME #{self.t-1} - **** TRACKER failure with score {tr_score}')

                self.id_counter += 1
                self.found = False
                
                
                return False, None, None, None
            else:
                # give some time to thracker to resume the tracking
                self.track_fail_counter += 1

                if self.t % self.save_last_box_every == 0:
                    self.last_box = box

                return True, self.id_counter, np.copy(box), tr_score

    def next_point(self, pts):
    
        x_diff = pts[-1, 0] - pts[-2, 0] 
        y_diff = pts[-1, 1] - pts[-2, 1]

        next_point = np.array([pts[-1, 0] + x_diff, pts[-1, 1] + y_diff])

        return next_point


