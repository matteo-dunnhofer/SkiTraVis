import os
import os.path
import torch
import numpy as np
import pandas
import csv
import random
from collections import OrderedDict
from .base_video_dataset import BaseVideoDataset
from lib.train.data import jpeg4py_loader
from lib.train.admin import env_settings
from lib.utils.box_ops import box_iou, box_xywh_to_xyxy
import json


class SkiTDST(BaseVideoDataset):
    """ LaSOT dataset.

    Publication:
        LaSOT: A High-quality Benchmark for Large-scale Single Object Tracking
        Heng Fan, Liting Lin, Fan Yang, Peng Chu, Ge Deng, Sijia Yu, Hexin Bai, Yong Xu, Chunyuan Liao and Haibin Ling
        CVPR, 2019
        https://arxiv.org/pdf/1809.07845.pdf

    Download the dataset from https://cis.temple.edu/lasot/download.html
    """

    def __init__(self, root=None, image_loader=jpeg4py_loader, vid_ids=None, split=None, data_fraction=None, disciplines=['AL', 'JP', 'FS'], split_mode='date'):
        """
        args:
            root - path to the lasot dataset.
            image_loader (jpeg4py_loader) -  The function to read the images. jpeg4py (https://github.com/ajkxyz/jpeg4py)
                                            is used by default.
            vid_ids - List containing the ids of the videos (1 - 20) used for training. If vid_ids = [1, 3, 5], then the
                    videos with subscripts -1, -3, and -5 from each class will be used for training.
            split - If split='train', the official train split (protocol-II) is used for training. Note: Only one of
                    vid_ids or split option can be used at a time.
            data_fraction - Fraction of dataset to be used. The complete dataset is used by default
        """
        self.root = env_settings().skitdlt_dir if root is None else root
        super().__init__('SkiTDST', root, image_loader)

        # Keep a list of all classes
        #self.class_list = [f for f in os.listdir(self.root)]
        #self.class_to_id = {cls_name: cls_id for cls_id, cls_name in enumerate(self.class_list)}
        self.disciplines = disciplines
        self.split_mode = split_mode

        #self.split_mode = 'course'

        self.sequence_list = self._build_sequence_list(vid_ids, split)

        #if data_fraction is not None:
        #    self.sequence_list = random.sample(self.sequence_list, int(len(self.sequence_list)*data_fraction))

        #self.seq_per_class = self._build_class_list()

    def _build_sequence_list(self, vid_ids=None, split=None):

        sequence_list = []
        for discipline_id in self.disciplines:
            disc_dir = os.path.join(self.root, discipline_id)
            print(f'Dataset split {split} - file: {discipline_id}_train_test_{self.split_mode}_60-40.json')
            split_file = os.path.join(disc_dir, f'{discipline_id}_train_test_{self.split_mode}_60-40.json')
            fs = open(split_file)
            split_dict = json.load(fs)
            if split == 'val':
                vid_names = split_dict['train']
            else:
                vid_names = split_dict[split]

            n_val_vids = round(0.1 * len(vid_names))

            if split == 'train':
                vid_names = vid_names[:-n_val_vids]
            elif split == 'val':
                vid_names[-n_val_vids:]


            if discipline_id == 'AL':
                valid_idxs = []
                for i, test_seq in enumerate(vid_names):
                    if os.path.exists(os.path.join(self.root, 'AL', test_seq, 'ST')):
                        valid_idxs.append(i)

                vid_names = [vid_names[vi] for vi in valid_idxs]

            for vid in vid_names:
                st_sequences = os.listdir(os.path.join(self.root, discipline_id, vid, 'ST'))
                sequence_list += [f'{vid}/{st}' for st in st_sequences]

        """
        if split is not None:
            if vid_ids is not None:
                raise ValueError('Cannot set both split_name and vid_ids.')
            ltr_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
            if split == 'train':
                file_path = os.path.join(ltr_path, 'data_specs', 'lasot_train_split.txt')
            else:
                raise ValueError('Unknown split name.')
            sequence_list = pandas.read_csv(file_path, header=None, squeeze=True).values.tolist()
        elif vid_ids is not None:
            sequence_list = [c+'-'+str(v) for c in self.class_list for v in vid_ids]
        else:
            raise ValueError('Set either split_name or vid_ids.')
        """
        return sequence_list

    """
    def _build_class_list(self):
        seq_per_class = {}
        for seq_id, seq_name in enumerate(self.sequence_list):
            class_name = seq_name.split('-')[0]
            if class_name in seq_per_class:
                seq_per_class[class_name].append(seq_id)
            else:
                seq_per_class[class_name] = [seq_id]

        return seq_per_class
    """

    def get_name(self):
        return 'skitdst'

    def has_class_info(self):
        return False

    def has_occlusion_info(self):
        return True

    def get_num_sequences(self):
        return len(self.sequence_list)

    def get_num_classes(self):
        return 1

    #def get_sequences_in_class(self, class_name):
    #    return self.seq_per_class[class_name]

    def _read_bb_anno(self, seq_path):
        bb_anno_file = os.path.join(seq_path, "boxes.txt")
        gt = pandas.read_csv(bb_anno_file, delimiter=',', header=None, dtype=np.float32, na_filter=False, low_memory=False).values
        return torch.tensor(gt)

    def _read_target_visible(self, seq_path):
        # Read full occlusion and out_of_view
        occlusion_file = os.path.join(seq_path, "visibilities.txt")
        #out_of_view_file = os.path.join(seq_path, "out_of_view.txt")

        #with open(occlusion_file, 'r', newline='\n') as f:
        #    occlusion = torch.ByteTensor([int(v) for v in list(csv.reader(f))[0]])
        #with open(out_of_view_file, 'r') as f:
        #    out_of_view = torch.ByteTensor([int(v) for v in list(csv.reader(f))[0]])
        target_visible = torch.ByteTensor(np.loadtxt(occlusion_file, delimiter='\n'))

        #target_visible = occlusion #& ~out_of_view
        #print(target_visible)

        return target_visible

    def _get_sequence_path(self, seq_id):
        seq_name = self.sequence_list[seq_id].split('/')[0]
        seq_id_str = self.sequence_list[seq_id].split('/')[1]
        disc_id = seq_name[:2]
        #vid_id = seq_name.split('-')[1]
        return os.path.join(self.root, disc_id, seq_name, 'ST', seq_id_str)

    def get_sequence_info(self, seq_id):
        seq_path = self._get_sequence_path(seq_id)
        bbox = self._read_bb_anno(seq_path)

        valid = (bbox[:, 2] > 0) & (bbox[:, 3] > 0)
        visible = self._read_target_visible(seq_path) & valid.byte()

        return {'bbox': bbox, 'valid': valid, 'visible': visible}
    
    """
    def get_sequence_info_random_box(self, seq_id, W, H):
        seq_path = self._get_sequence_path(seq_id)
        bbox = self._read_bb_anno(seq_path)

        random_boxes = []
        visibles = []
        for bi in range(bbox.shape[0]):
            x = random.randint(0, W-1)
            y = random.randint(0, H-1)
            w = random.randint(4, W-1)
            h = random.randint(4, H-1)

            random_box = [x,y,w,h]
            visible = box_iou(box_xywh_to_xyxy([random_box]), box_xywh_to_xyxy([bbox[bi]]))[0] >= 0.7

            random_boxes.append(random_box)
            visibles.append(visible)

        bbox = torch.tensor(random_box)
        visibles = torch.tensor(visibles)

        valid = (bbox[:, 2] > 0) & (bbox[:, 3] > 0)
        #visible = self._read_target_visible(seq_path) & valid.byte()

        return {'bbox': bbox, 'valid': valid, 'visible': visibles}
    """

    def _get_frame_path(self, seq_path, frame_id):
        #print(seq_path[:-1].split('/')[-1])
        
        seq_name = seq_path[:-1].split('/')[-2]
        #print(seq_name)
        disc_id = seq_name[:2]
        vid_num = int(seq_name[2:])

        lt_seq_path = os.path.join(self.root, disc_id, seq_name)

        if disc_id == 'AL' and vid_num <= 30:
            return os.path.join(lt_seq_path, 'frames-all-ext', '{:05d}.jpg'.format(frame_id))    # frames start from 1
        else:
            return os.path.join(lt_seq_path, 'frames-all', '{:05d}.jpg'.format(frame_id))    # frames start from 1

    def _get_frame(self, seq_path, frame_id):
        return self.image_loader(self._get_frame_path(seq_path, frame_id))

    #def _get_class(self, seq_path):
    #    raw_class = seq_path.split('/')[-2]
    #    return raw_class

    #def get_class_name(self, seq_id):
    #    seq_path = self._get_sequence_path(seq_id)
    #    obj_class = self._get_class(seq_path)
    #
    #    return obj_class

    def get_frames(self, seq_id, frame_ids, anno=None):
        seq_path = self._get_sequence_path(seq_id)

        frames_file = os.path.join(seq_path, "frames.txt")
        frame_idxs = np.loadtxt(frames_file, delimiter='\n')

        #print(frame_ids, len(frame_idxs))

        #assert len(frame_ids) == len(frame_idxs)
        #obj_class = self._get_class(seq_path)
        frame_list = [self._get_frame(seq_path[:-2], int(frame_idxs[f_id])) for f_id in frame_ids]
        #print(frame_list)

        if anno is None:
            anno = self.get_sequence_info(seq_id)

        anno_frames = {}
        for key, value in anno.items():
            anno_frames[key] = [value[f_id, ...].clone() for f_id in frame_ids]

        object_meta = OrderedDict({'object_class_name': 'skier',
                                   'motion_class': None,
                                   'major_class': None,
                                   'root_class': None,
                                   'motion_adverb': None})

        return frame_list, anno_frames, object_meta
    
    def get_frames_random_box(self, seq_id, frame_ids, anno=None):
        seq_path = self._get_sequence_path(seq_id)

        frames_file = os.path.join(seq_path, "frames.txt")
        frame_idxs = np.loadtxt(frames_file, delimiter='\n')

        #print(frame_ids, len(frame_idxs))

        #assert len(frame_ids) == len(frame_idxs)
        #obj_class = self._get_class(seq_path)
        frame_list = [self._get_frame(seq_path[:-2], int(frame_idxs[f_id])) for f_id in frame_ids]
        #print(frame_list)

        if anno is None:
            anno = self.get_sequence_info_random_box(seq_id)

        anno_frames = {}
        for key, value in anno.items():
            anno_frames[key] = [value[f_id, ...].clone() for f_id in frame_ids]

        object_meta = OrderedDict({'object_class_name': 'skier',
                                   'motion_class': None,
                                   'major_class': None,
                                   'root_class': None,
                                   'motion_adverb': None})

        return frame_list, anno_frames, object_meta
