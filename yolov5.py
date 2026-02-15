import torch
from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh, xyxy2xywh_tl)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode

import numpy as np



class YOLOv5(object):

    def __init__(self, weights, device):

        self.imgsz=(640, 640)
        self.conf_thres = 0.75 #0.25
        self.iou_thres = 0.45
        self.agnostic_nms = False
        self.classes = None #0
        self.max_det = 10 #1000
        self.augment = False
        self.data = None
        self.half = False
        self.dnn = False

        self.model = DetectMultiBackend(weights, device=device, dnn=self.dnn, data=self.data, fp16=self.half)

        self.dt = (Profile(), Profile(), Profile())

    
    def warmup(self, imgsz, bs=1):
        self.model.warmup(imgsz=(1 if self.model.pt or self.model.triton else bs, 3, *imgsz))  # warmup

    
    def detect(self, im):
        """
        Get YOLO skier detections for an image
        """
        #
        im = np.copy(im).transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(im)  # contiguous

        with self.dt[0]:
            im = torch.from_numpy(im).to(self.model.device)
            im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with self.dt[1]:
            #visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = self.model(im, augment=self.augment, visualize=False)

        # NMS
        with self.dt[2]:
            pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, max_det=self.max_det)

        det = pred[0]

        if len(det):
            
            max_conf_idx = torch.argmax(det[:, 4])

            box = det[max_conf_idx, :4]
            box = xyxy2xywh_tl(box.view(1, 4)).view(-1).cpu().numpy()

            score = det[max_conf_idx, 4].item()

            return box, score
        else:
            # no detections given
            return None, None


    def detect_all(self, im):
        """
        Get YOLO skier detections for an image
        """
        #
        im = np.copy(im).transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(im)  # contiguous

        with self.dt[0]:
            im = torch.from_numpy(im).to(self.model.device)
            im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with self.dt[1]:
            pred = self.model(im, augment=self.augment, visualize=False)

        # NMS
        with self.dt[2]:
            pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, max_det=self.max_det)

        det = pred[0]

        return det

        if len(det):
            
            return det #box, score
        else:
            # no detections given
            return None


      
    def detect_(self, im, im0s, webcam):
        """
        Get YOLO skier detections for an image
        """
        #
        with self.dt[0]:
            im = torch.from_numpy(im).to(self.model.device)
            im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with self.dt[1]:
            pred = self.model(im, augment=self.augment, visualize=False)

        # NMS
        with self.dt[2]:
            pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, max_det=self.max_det)

        
        det = pred[0]

        if webcam:  # batch_size >= 1
            im0 = im0s[i].copy()
        
        else:
            im0 = im0s.copy()

        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]

        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

            max_conf_idx = torch.argmax(det[:, 4])

            box = det[max_conf_idx, :4]
            score = det[max_conf_idx, 4]

            return box, score
        else:
            # no detections given
            return None, None


        # Process predictions
        for i, det in enumerate(pred):  # per image

            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(f'{txt_path}.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)