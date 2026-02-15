import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from dataset import SkiTopDownDataset
import utils
import transforms as T
from engine import train_one_epoch, evaluate
import argparse

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

parser = argparse.ArgumentParser()
parser.add_argument('--disciplines', help='Disciplines to download', nargs='+', type=str, default='ALL')
parser.add_argument('--split', help='Condition for the split (athlete, course, date)', type=str, default='date')
parser.add_argument('--model', help='Name for the model to test (FasterRCNN-res50, FasterRCNN-mob, RetinaNet-res50, SSDLite)', type=str, default='FasterRCNN-res50')
parser.add_argument('--ckpt', help='Path to saved weight file', type=str, default='date')
args = parser.parse_args()

if len(args.disciplines) == 1 and args.disciplines[0] == 'ALL':
    disciplines = ['AL', 'JP', 'FS']
else:
    disciplines = args.disciplines

# replace the classifier with a new one, that has
# num_classes which is user-defined
num_classes = 2  # 1 class (person) + background

# load a model pre-trained on COCO
if args.model == 'FasterRCNN-res50':
    weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights=weights)
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    ft_weights = torch.load(args.ckpt)
    model.load_state_dict(ft_weights)
elif args.model == 'FasterRCNN-mob':
    weights = torchvision.models.detection.FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.COCO_V1
    model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(weights=weights)
elif args.model == 'RetinaNet-res50':
    weights = torchvision.models.detection.RetinaNet_ResNet50_FPN_V2_Weights.COCO_V1
    model = torchvision.models.detection.retinanet_resnet50_fpn_v2(weights=weights)
elif args.model == 'SSDLite':
    weights = torchvision.models.detection.SSDLite320_MobileNet_V3_Large_Weights.COCO_V1
    model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(weights=weights)


# train on the GPU or on the CPU, if a GPU is not available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# our dataset has two classes only - background and person
num_classes = 2

# use our dataset and defined transformations
dataset_root = '/home/matteo/Desktop/datasets/SkiVideoNet/dataset'
dataset_test = SkiTopDownDataset(dataset_root, args.disciplines, split=args.split, train=False, transforms=get_transform(train=False))

print(f'Number of test images: {len(dataset_test)}')

data_loader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=1, shuffle=False, num_workers=4,
    collate_fn=utils.collate_fn)

# move model to the right device
model.to(device)

# let's train it for 10 epochs
num_epochs = 10

evaluate(model, data_loader_test, device=device)
