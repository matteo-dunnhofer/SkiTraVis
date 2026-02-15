import sys
sys.path.append('./tracking/Stark')

import os
env_path = os.path.join(os.path.dirname(__file__), '..')
if env_path not in sys.path:
    sys.path.append(env_path)

from lib.test.evaluation import Tracker as Stark

import numpy as np
import torch
import cv2

class STARK(object):

    def __init__(self, ckpt_path=None):
        tracker_info = Stark('stark_st', 'baseline_SkiTD_JP-FS-AL', "otb", None)
        params = tracker_info.get_parameters()
        params.visualization = False
        params.debug = False
        params.checkpoint = ckpt_path 
        self.stark = tracker_info.create_tracker(params)
        #self.stark.params.search_factor = 2.5

    def init(self, image, box):

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        box = [int(box[0]), int(box[1]), int(box[2]), int(box[3])]
        init_info = {'init_bbox': box}
        self.stark.initialize(image, init_info)

    def update(self, image):
        """
        Given current image, returns target box.
        """
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        outputs = self.stark.track(image)
        pred_bbox = outputs['target_bbox']
        box = np.array(pred_bbox)
        conf = outputs['conf_score']
        
        return box, conf
