
import argparse
import os
import platform
import sys
sys.path.append('./detection/yolov5')
sys.path.append('./tracking/Stark')

from pathlib import Path

import fcntl

import torch

"""
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
"""

from models.common import DetectMultiBackend
from utils.dataloaders2 import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode

import numpy as np
from scipy.signal import savgol_filter

from kornia.geometry.ransac import RANSAC

import time
from yolov5 import YOLOv5
from stark import STARK
from dettrack import DetTrack
from superglue import SuperGlue
import pydegensac
from loftr import LOFTR

import matplotlib.pyplot as plt


def smooth_trajectory_fun(traj, window_length=30, order=3):
    traj_x = savgol_filter(traj[:,0], window_length, order).reshape((traj.shape[0], 1))
    traj_y = savgol_filter(traj[:,1], window_length, order).reshape((traj.shape[0], 1))
    return np.concatenate((traj_x, traj_y), axis=1).astype(np.int64)

def ranges(nums):
    ranges = []
    start = 0
    end = -1

    for i in range(len(nums)-1):
        if nums[i+1] > nums[i] + 1:
            ranges.append((start, i))
            start = i+1

    ranges.append((start, len(nums)-1))

    return ranges

def build_kpt_filter_mask_from_boxes(boxes, mask=None, size=(640, 480)): #(w, h)

    if mask is None:
        mask = np.zeros((size[1], size[0]), dtype=np.uint8)

    for i, box in enumerate(boxes):
        x, y, w, h = int(boxes[i][0]), int(boxes[i][1]), int(boxes[i][2]), int(boxes[i][3])
        mask[y:y+h, x:x+w] = 1

    return mask


def estimate_homography(matched_kpts1, matched_kpts2, method='pydegensac'):
    if method == 'ransac':
        homography, _ = cv2.findHomography(matched_kpts1, matched_kpts2)
    elif method == 'pydegensac':
        homography, _ = pydegensac.findHomography(matched_kpts1, matched_kpts2, 3.0, 0.99, 2000)
    elif method == 'kornia_ransac':
        homography, _ = RANSAC('homography')(torch.tensor(matched_kpts1), torch.tensor(matched_kpts2)) #(keypoints2, keypoints1)
        homography = homography.numpy()

    return homography


@smart_inference_mode()
def run(
        detector_weights='./checkpoints/YOLOv5x-AL.pt',  # model path or triton URL
        tracker_weights='./checkpoints/STARKST_AL.pth.tar',
        matcher='superglue',
        source='data/images',  # file/dir/URL/glob/screen/0(webcam)
        imgsz=(640, 640),  # inference size (height, width)
        track_yolo_res=False,
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_dir=None,
        debug=True,
        line_thickness=3,
        feet_position_factor=0.9,
        prev_traj_type='minmax', # last or minmax (based on running time)
        smooth_trajectory=True,
        trajectory_smooth_window=20,
        trajectory_smooth_order=3,
        update_prev_data_every=1,
        draw_kpt_filter_mask=False,
        athlete_name='Athlete',
        draw_logos=False,
        manual_init=False
):
    source = str(source)
    #save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    screenshot = source.lower().startswith('screen')
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    if (save_dir is not None):
        save_path = os.path.join(save_dir, f'{source.split(os.sep)[-1].split(".")[0]}_skitravis.mp4')

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    
    # Load YOLO model
    device = select_device(device)

    yolo = YOLOv5(detector_weights, device)
    stark = STARK(tracker_weights)

    dettrack = DetTrack(yolo, stark, verbose=True)
    
    stride, names, pt = yolo.model.stride, yolo.model.names, yolo.model.pt
    
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    bs = 1  # batch_size
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
    vid_path, vid_writer = [None], [None]

    # Run inference
    dettrack.warmup(imgsz)


    if matcher == 'superglue':
        matcher = SuperGlue(device=device, verbose=True)
    elif matcher == 'loftr':
        matcher = LOFTR(device=device, verbose=True)
    homography_estimation_method = 'pydegensac'

    seen, windows = 0, []

    colors = {'green' : (108, 184, 116), 
                'red' : (58, 58, 222),
                'grey' : (210, 210, 210),
                'blue' : (204, 137, 70)}

    

    trajectories_dict = {}
    trajectory_lengths_dict = {}
    trajectory_idxs = {}

    tracking = False

    current_target_id = -1
    last_target_id = -1
    traj_computed = False
    
    
    f_idx = 0

    if draw_logos:
        logo2 = cv2.imread('mlp_text_uniud.png')

    for path, im, im0s, vid_cap, s in dataset:

        if manual_init and f_idx == 0:
            print('Click on the top-left and drag to the bottom-right corners of the bounding-box around the skier to initialize the tracker, then press ENTER or SPACE. Press ESC to cancel the bounding-box selection.')

            cv2.namedWindow("Draw bounding-box over skier for tracker initialization", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Draw bounding-box over skier for tracker initialization", im0s.shape[1], im0s.shape[0])
            
            try:
                init_rect = cv2.selectROI('Draw bounding-box over skier for tracker initialization', im0s, False, False)
                x, y, w, h = init_rect
                init_box = np.array([x, y, x+w, y+h]).astype(np.float32)
                init_box = scale_boxes(im0s.shape[:2], np.expand_dims(np.copy(init_box), 0), im.shape).round()[0]
                init_box = [init_box[0], init_box[1], init_box[2]-init_box[0], init_box[3]-init_box[1]]
                cv2.destroyAllWindows()
            except Exception as e:
                print(e)
                exit()
        else:
            init_box = None

        if draw_kpt_filter_mask and f_idx == 0:
            print('Click on the top-left and drag to the bottom-right corners of the bounding-box around the areas to exclude keypoints for the homography estimation (e.g. superimposed logos). Press SPACE to confirm a selection and move to the next. Press ENTER to finish and run processing.')

            cv2.namedWindow("Draw bounding-boxes to filter keypoints", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Draw bounding-boxes to filter keypoints", im0s.shape[1], im0s.shape[0])
            
            try:
                static_kpt_filtering_boxes = cv2.selectROIs('Draw bounding-boxes to filter keypoints', im0s, False, False)
                static_kpt_filtering_boxes = np.array(static_kpt_filtering_boxes)
                
                cv2.destroyAllWindows()

                
                static_kpt_filtering_mask = build_kpt_filter_mask_from_boxes(static_kpt_filtering_boxes, size=(im0s.shape[1], im0s.shape[0]))

                print(im0s.shape)
            except Exception as e:
                print(e)
                exit()
        else:
            static_kpt_filtering_mask = build_kpt_filter_mask_from_boxes([], size=(im0s.shape[1], im0s.shape[0]))

    
        if track_yolo_res:
            tracking, target_id, target_bbox, score = dettrack.update(det_frame=im, track_frame=im, init_box=init_box)
            im = im
        else:
            tracking, target_id, target_bbox, score = dettrack.update(det_frame=im, track_frame=im0s, init_box=init_box)
            im = im0s

        if tracking:
            target_bbox[2:] = target_bbox[:2] + target_bbox[2:]
            target_bbox = scale_boxes(im.shape[:2], np.expand_dims(target_bbox, 0), im0s.shape).round()[0]
           

        x1_p = target_bbox[0] - 0.1 * (target_bbox[2] - target_bbox[0])
        y1_p = target_bbox[1] - 0.1 * (target_bbox[3] - target_bbox[1])
        kpt_filtering_mask = build_kpt_filter_mask_from_boxes([[x1_p, y1_p, target_bbox[2] + 0.1 * (target_bbox[2] - target_bbox[0]) - x1_p, target_bbox[3] + 0.1 * (target_bbox[3] - target_bbox[1]) - y1_p]], mask=np.copy(static_kpt_filtering_mask))
        
        if f_idx == 0:
            matcher.set_prev_frame(im0s)

            homography = np.eye(3)
        else:
            
            mkpts0, mkpts1, sg_data = matcher.match_w_prev(im0s, filter_mask=kpt_filtering_mask)

            if f_idx % update_prev_data_every == 0:
                matcher.set_prev_data(sg_data)

            st_time = time.time()

            homography = estimate_homography(mkpts0, mkpts1, method=homography_estimation_method)

            hg_time = time.time() - st_time

            if debug:
                print(f'HOMOGRAPHY computed in {np.round(hg_time, 3)}s')
            

        im0 = im0s.copy()
        
        if draw_kpt_filter_mask:
            for g_box in static_kpt_filtering_boxes:
                x,y,w,h = int(g_box[0]), int(g_box[1]), int(g_box[2]), int(g_box[3])
                im0[y:y+h, x:x+w] = cv2.blur(np.copy(im0[y:y+h, x:x+w]), (30, 30))

        st_time = time.time()

        annotator = Annotator(im0, line_width=line_thickness)
        
        if debug:
            annotator.text_cv2((20, 40), f'{f_idx:05d}')
            if use_det_area:
                annotator.box_label(det_area_box.round(), label=f'Detection area', color=(255, 0, 0))

            if f_idx > 0:
                annotator.circles(mkpts1.astype(np.int64), color=colors['red'])


        if tracking:

            current_target_id = target_id
            
            traj_computed = False

            bc_x = (target_bbox[0] + target_bbox[2]) / 2
            bc_y = target_bbox[1] + (target_bbox[3] - target_bbox[1]) * feet_position_factor

            if debug:
                annotator.points([[int(bc_x), int(bc_y)]], color=colors['red'])

            if not target_id in trajectories_dict:
                trajectories_dict[target_id] = [[bc_x, bc_y]]
                trajectory_lengths_dict[target_id] = 1
                trajectory_idxs[target_id] = [f_idx]
            else:
                trajectories_dict[target_id].append([bc_x, bc_y])
                trajectory_lengths_dict[target_id] += 1
                trajectory_idxs[target_id].append(f_idx)

        
            if len(trajectories_dict[target_id]) > 1:


                bc_traj_pts = np.array(trajectories_dict[target_id])[:-1]
                
                # apply homography
                bc_traj_pts = bc_traj_pts.reshape(-1,1,2).astype(np.float32)
                bc_traj_pts = cv2.perspectiveTransform(bc_traj_pts, homography).reshape((-1, 2))
                
                # filter visible points
                cond = (0 <= bc_traj_pts[:,0]) * (bc_traj_pts[:,0] < im.shape[1]) * (0 <= bc_traj_pts[:,1]) * (bc_traj_pts[:,1] < im.shape[0])
                bc_traj_pts = bc_traj_pts[cond]

                trajectories_dict[target_id] = bc_traj_pts.tolist() + [trajectories_dict[target_id][-1]]
                trajectory_idxs[target_id] = np.array(trajectory_idxs[target_id][:-1])[cond].tolist() + [f_idx]

                
                traj_sections = ranges(trajectory_idxs[target_id])

                for (start, end) in traj_sections:
                    bc_traj_pts_ = np.copy(bc_traj_pts[start:end+1])

                    if smooth_trajectory:
                        if bc_traj_pts_.shape[0] >= trajectory_smooth_window:
                            bc_traj_pts_ = smooth_trajectory_fun(bc_traj_pts_, trajectory_smooth_window, trajectory_smooth_order)

                    bc_traj_pts_ = np.int64(bc_traj_pts_)

                    cond1 = (target_bbox[0] <= bc_traj_pts_[:,0]) * (bc_traj_pts_[:,0] <= target_bbox[2]) * (target_bbox[1] <= bc_traj_pts_[:,1]) * (bc_traj_pts_[:,1] <= target_bbox[3])
                    bc_traj_pts_ = bc_traj_pts_[~cond1]

                    annotator.trajectory(bc_traj_pts_, color=colors['blue'],  alpha=0.75) # rgb 75, 142, 209

                    if debug:
                        annotator.points(bc_traj_pts_, color=colors['blue'])

            annotator.top_label(((target_bbox[2] + target_bbox[0]) / 2,  target_bbox[1]), athlete_name, color=colors['blue'])
        
        else:
            last_target_id = current_target_id


        if draw_logos:
            logo2_x, logo2_y = im0.shape[1] - logo2.shape[1] - 50, 50 
            im0[logo2_y:logo2_y+logo2.shape[0], logo2_x:logo2_x+logo2.shape[1]] = np.uint8(im0[logo2_y:logo2_y+logo2.shape[0], logo2_x:logo2_x+logo2.shape[1]] * 0.3 + logo2 * 0.7)


        im0 = annotator.result()

        if view_img:
            if platform.system() == 'Linux' and path not in windows:
                windows.append(path)
                cv2.namedWindow(str(path), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                cv2.resizeWindow(str(path), im0.shape[1], im0.shape[0])
            cv2.imshow(str(path), im0)
            cv2.waitKey(1)  # 1 millisecond

        render_time = time.time() - st_time

        if debug:
            print(f'Visual rendering time {np.round(render_time, 3)}s')
        
        
        # Save results (image with detections)
        if save_dir is not None:
            if vid_path != save_path:  # new video
                vid_path = save_path
                if isinstance(vid_writer, cv2.VideoWriter):
                    vid_writer.release()  # release previous video writer
                if vid_cap:  # video
                    fps = vid_cap.get(cv2.CAP_PROP_FPS)
                    w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                else:  # stream
                    fps, w, h = 30, im0.shape[1], im0.shape[0]
                save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
            vid_writer.write(im0)

        
        f_idx += 1


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='./videos/lp4.mp4',
                        help='Input source: file, directory, URL, glob pattern, "screen" for screenshots, or camera index (e.g. "0").')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640],
                        help='Inference image size. Provide one value for square (H=W) or two values: H W.')
    parser.add_argument('--device', default='',
                        help='Computation device: "cpu", a CUDA index like "0", or multiple "0,1". Leave empty for auto selection.')
    parser.add_argument('--view-img', action='store_true',
                        help='Display annotated frames in a GUI window during processing.')
    parser.add_argument('--line-thickness', default=3, type=int,
                        help='Thickness in pixels for drawn boxes, lines and trajectory overlays.')
    parser.add_argument('--save-dir', default='./output',
                        help='Directory where output video and images will be written.')
    parser.add_argument('--debug', action='store_true',
                        help='Enable verbose debug prints and extra visual debugging overlays.')
    parser.add_argument('--smooth-trajectory', action='store_true',
                        help='Enable smoothing of computed trajectories using a Savitzky–Golay filter.')
    parser.add_argument('--feet-position-factor', type=float, default=0.9,
                        help='Relative vertical factor inside the bbox used to estimate the skier\'s feet position (0.0 - top, 1.0 - bottom).')
    parser.add_argument('--trajectory-smooth-window', type=int, default=20,
                        help='Window length (number of frames) for trajectory smoothing (should be odd for Savitzky–Golay).')
    parser.add_argument('--trajectory-smooth-order', type=int, default=3,
                        help='Polynomial order for the Savitzky–Golay trajectory smoother.')
    parser.add_argument('--athlete-name', type=str, default='Athlete',
                        help='Text label to display above the tracked athlete.')
    parser.add_argument('--manual-init', action='store_true',
                        help='Require manual initialization: draw a bounding box in the first frame to initialize the tracker.')
    

    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
