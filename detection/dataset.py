import os
import numpy as np
import torch
import torch.utils.data
from PIL import Image
import json


class SkiTopDownDataset(torch.utils.data.Dataset):

    def __init__(self, root, vids, disciplines=['AL', 'JP', 'FS'], split='date', train=True, transforms=None):
        self.root = root
        self.vids = vids
        self.transforms = transforms
        self.disciplines = disciplines
        self.split = split
        self.train = train
        # load all image files, sorting them to
        # ensure that they are aligned
        self._load_data()

    def _load_data(self):

        self.img_paths = []
        self.boxes = []
        
        for discipline_id in self.disciplines:
            print(f'Loading data for discipline {discipline_id}')

        #disc_dir = os.path.join(self.root, discipline_id)
        #seq_names = os.listdir(disc_dir)
        #seq_names = [filename for filename in os.listdir(disc_dir) if os.path.isdir(os.path.join(disc_dir, filename))]

        #print(f'Dataset split {self.split} - file: {discipline_id}_train_test_{self.split}.json')
        #split_file = os.path.join(disc_dir, f'{discipline_id}_train_test_{self.split}.json')
        #fs = open(split_file)
        #split_dict = json.load(fs)

        #if self.train:
        #    seq_names = split_dict['train']
        #else:
        #    seq_names = split_dict['test']

        #for seq_name in seq_names:
        for seq_name in self.vids:

            #print(seq_name)
            discipline_id = seq_name[:2]
            disc_dir = os.path.join(self.root, discipline_id)

            seq_path = os.path.join(disc_dir, seq_name)

            f = open(os.path.join(seq_path, 'box-annotations.json'))
            anno_dict = json.load(f)

            for frame_idx in anno_dict.keys():

                bbox = anno_dict[frame_idx]['bbox']
                bbox_occluded = anno_dict[frame_idx]['bbox_occluded']

                if bbox_occluded == 0: # if not occluded
                    self.img_paths.append(os.path.join(seq_path, 'frames-all', f'{int(frame_idx):05d}.jpg'))
                    self.boxes.append(bbox)

            break


    def __getitem__(self, idx):
        # load images ad masks
        img_path = self.img_paths[idx]
        img = Image.open(img_path).convert("RGB")
        
        box = self.boxes[idx]
        box[2] = box[0] + box[2]
        box[3] = box[1] + box[3]
        boxes = [box]
        num_objs = len(boxes)

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.img_paths)