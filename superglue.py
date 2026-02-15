import sys
sys.path.append('./SuperGluePretrainedNetwork')

from pathlib import Path
import argparse
import cv2
import matplotlib.cm as cm
import torch
import numpy as np
import pydegensac
import os
import time

from SuperGluePretrainedNetwork.models.matching import Matching
from SuperGluePretrainedNetwork.models.utils import (AverageTimer, VideoStreamer,
                          make_matching_plot_fast, frame2tensor)


class SuperGlue(object):

    def __init__(self, device=None, ckpt_path=None, verbose=False):
        config = {
            'superpoint': {
                'nms_radius': 4,
                'keypoint_threshold': 0.005,
                'max_keypoints': -1 
            },
            'superglue': {
                'weights': 'outdoor',
                'sinkhorn_iterations': 20,
                'match_threshold': 0.1,
            }
        }
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() and not opt.force_cpu else 'cpu'
        else:
            self.device = device
        self.matching = Matching(config).eval().to(self.device)
        self.keys = ['keypoints', 'scores', 'descriptors']

        self.verbose = verbose

    def set_prev_frame(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_tensor = frame2tensor(frame, self.device)
        with torch.no_grad():
            self.last_data = self.matching.superpoint({'image': frame_tensor})
        self.last_data = {k+'0': self.last_data[k] for k in self.keys}
        self.last_data['image0'] = frame_tensor
        #self.last_frame = frame
        self.last_image_id = 0

    def set_prev_data(self, data):
        self.last_data = {k+'0': data[k+'1'] for k in self.keys}
        self.last_data['image0'] = data['image1'] 
        #self.last_frame = frame
        self.last_image_id = 0

    def match_w_prev(self, frame, filter_mask=None):
        """
        Given current image, returns target box.
        """
        st_time = time.time()

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_tensor = frame2tensor(frame, self.device)

        with torch.no_grad():
            new_data = self.matching.superpoint({'image': frame_tensor})
        
        new_data = {k+'1': new_data[k] for k in self.keys}
        new_data['image1'] = frame_tensor

        with torch.no_grad():
            pred = self.matching({**new_data, **self.last_data})
        kpts0 = self.last_data['keypoints0'][0].cpu().numpy()
        kpts1 = new_data['keypoints1'][0].cpu().numpy()
        matches = pred['matches1'][0].cpu().numpy()
        confidence = pred['matching_scores0'][0].detach().cpu().numpy()

        if filter_mask is not None:
            matches = self.filter_matches(kpts1, matches, filter_mask)

        valid = matches > -1
        mkpts0 = kpts0[matches[valid]]
        mkpts1 = kpts1[valid]

        matching_time = time.time() - st_time

        if self.verbose:
            print(f'SUPERGLUE MATCHING run in {np.round(matching_time, 3)}s')
        
        return mkpts0, mkpts1, new_data

    def match_images(self, frame_src, frame_dst, filter_mask1=None, filter_mask2=None):
        """
        Given current image, returns target box.
        """
        st_time = time.time()

        frame_src = cv2.cvtColor(frame_src, cv2.COLOR_BGR2GRAY)
        frame_tensor1 = frame2tensor(frame_src, self.device)

        new_data = self.matching.superpoint({'image': frame_tensor1})

        new_data = {k+'0': new_data[k] for k in self.keys}
        new_data['image0'] = frame_tensor1

        frame_dst = cv2.cvtColor(frame_dst, cv2.COLOR_BGR2GRAY)
        frame_tensor2 = frame2tensor(frame_dst, self.device)

        prev_data = self.matching.superpoint({'image': frame_tensor2})
    
        prev_data = {k+'1': prev_data[k] for k in self.keys}
        prev_data['image1'] = frame_tensor2

    
        pred = self.matching({**new_data, **prev_data})
        kpts0 = new_data['keypoints0'][0].cpu().numpy()
        kpts1 = prev_data['keypoints1'][0].cpu().numpy()
        matches = pred['matches0'][0].cpu().numpy()
        confidence = pred['matching_scores0'][0].cpu().numpy()

        if filter_mask1 is not None:
            matches = self.filter_matches(kpts0, matches, filter_mask1)
        if filter_mask2 is not None:
            matches = self.filter_matches(kpts0, matches, filter_mask2)

        valid = matches > -1
        mkpts0 = kpts0[valid]
        mkpts1 = kpts1[matches[valid]]

        matching_time = time.time() - st_time

        if self.verbose:
            print(f'SUPERGLUE MATCHING run in {np.round(matching_time, 3)}s')
        
        return mkpts0, mkpts1

    def filter_kpts(self, data, filter_mask):

        kpts = data['keypoints'][0].cpu().numpy()
        descs = data['descriptors'][0].cpu().numpy()
        scores = data['scores'][0].cpu().numpy()

        kpts = np.fliplr(kpts)
        kpts_mask = np.zeros_like(filter_mask)
        kpts_mask[np.int64(kpts[:,0]),np.int64(kpts[:,1])] = 1
        and_mask = (1 - filter_mask) * kpts_mask

        idx_mask = np.zeros_like(filter_mask)
        idx_mask[np.int64(kpts[:,0]),np.int64(kpts[:,1])] = np.arange(kpts.shape[0])

        idxs = np.where(and_mask > 0)
        idxs = np.concatenate((np.reshape(idxs[0], (-1, 1)), np.reshape(idxs[1], (-1, 1))), axis=1)

        k_idxs = idx_mask[np.int64(idxs[:,0]), np.int64(idxs[:,1])]
        new_kpts = np.fliplr(idxs).copy()

        new_kpts = np.fliplr(idxs).copy()
        new_descs = descs[:, k_idxs].copy()
        new_scores = scores[k_idxs].copy()
        new_data = {'keypoints' : [torch.tensor(new_kpts).to(self.device)], 
                    'descriptors' : [torch.tensor(new_descs).to(self.device)],  
                    'scores' : [torch.tensor(new_scores).to(self.device)] }

        return new_data

    def filter_matches(self, kpts, matches, filter_mask):
        new_matches = np.zeros_like(matches)
        for i, mi in enumerate(matches):
            if filter_mask[int(kpts[i,1]), int(kpts[i,0])] == 1:
                new_matches[i] = -1
            else:
                new_matches[i] = mi

        return new_matches
