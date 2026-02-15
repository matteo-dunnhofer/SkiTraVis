import os
import sys
sys.path.append('/media/TBData2/projects/SkiVideoNet/tracking/Stark')

env_path = os.path.join(os.path.dirname(__file__), '..')
if env_path not in sys.path:
    sys.path.append(env_path)

import numpy as np
from toolkit.got10k.trackers import Tracker

#from lib.test.tracker.stark_st import STARK_ST
from lib.test.evaluation import Tracker as TrackerMy


class TrackerStark(Tracker):
    """GOTURN got10k class for benchmark and evaluation on tracking datasets.

    This class overrides the default got10k 'Tracker' class methods namely,
    'init' and 'update'.

    Attributes:
        cuda: flag if gpu device is available
        device: device on which GOTURN is evaluated ('cuda:0', 'cpu')
        net: GOTURN pytorch model
        prev_box: previous bounding box
        prev_img: previous tracking image
        transform_tensor: normalizes images and returns torch tensor.
        otps: bounding box config to unscale and uncenter network output.
    """
    def __init__(self, tname='STARK', gpu_id=0, ckpt=None, **kargs):
        super(TrackerStark, self).__init__(
            name=tname, is_deterministic=True)

        #self.tracker = STARK_ST()
        tracker_info = TrackerMy('stark_st', 'baseline', "otb", None)
        params = tracker_info.get_parameters()
        params.visualization = False
        params.debug = False
        params.checkpoint = ckpt
        self.tracker = tracker_info.create_tracker(params)

    def init(self, image, box, **kwargs):
        """
        Initiates the tracker at given box location.
        Aassumes that the initial box has format: [xmin, ymin, width, height]
        """
        frame = np.array(image)

        box = [int(box[0]), int(box[1]), int(box[2]), int(box[3])]
        init_info = {'init_bbox': box}
        
        self.tracker.initialize(frame, init_info)

    def update(self, image, **kwargs):
        """
        Given current image, returns target box.
        """
        frame = np.array(image)

        outputs = self.tracker.track(frame)
        pred_bbox = outputs['target_bbox']
        box = np.array(pred_bbox)
        #print(outputs['conf_score'])

        return_conf = True
        if return_conf:
            conf = np.array(outputs['conf_score'])
            return box, conf
        else:
            return box
