import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from dataset import SkiTopDownDataset
import utils
import transforms as T
from engine import train_one_epoch, evaluate
import argparse
import os
import json

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
elif args.model == 'FasterRCNN-mob':
    weights = torchvision.models.detection.FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
    model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(weights=weights)
elif args.model == 'RetinaNet-res50':
    weights = torchvision.models.detection.RetinaNet_ResNet50_FPN_V2_Weights.COCO_V1
    model = torchvision.models.detection.retinanet_resnet50_fpn_v2(weights=weights)
elif args.model == 'SSDLite':
    weights = torchvision.models.detection.SSDLite320_MobileNet_V3_Large_Weights.COCO_V1
    model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(weights=weights)


# train on the GPU or on the CPU, if a GPU is not available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


# use our dataset and defined transformations
dataset_root = '/home/matteo/Desktop/datasets/SkiVideoNet/dataset'
train_vids = []
val_vids = []
for discipline_id in args.disciplines:
    disc_dir = os.path.join(dataset_root, discipline_id)
    print(f'Dataset split {args.split} - file: {discipline_id}_train_test_{args.split}.json')
    split_file = os.path.join(disc_dir, f'{discipline_id}_train_test_{args.split}.json')
    fs = open(split_file)
    split_dict = json.load(fs)
    vid_names = split_dict['train']
    n_val_vids = round(0.1 * len(vid_names))

    train_vids += vid_names[:-n_val_vids]
    val_vids += vid_names[-n_val_vids:]

dataset = SkiTopDownDataset(dataset_root, train_vids, args.disciplines, split=args.split, train=True, transforms=get_transform(train=True))
dataset_val = SkiTopDownDataset(dataset_root, val_vids, args.disciplines, split=args.split, train=False, transforms=get_transform(train=False))

print(f'Number of training images: {len(dataset)}')

# split the dataset in train and test set
#indices = torch.randperm(len(dataset)).tolist()
#dataset = torch.utils.data.Subset(dataset, indices[:-50])
#dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

BATCH_SIZE = 4
LR = 0.005
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005
EPOCHS = 10

CKPT_PATH = './ckpt'
if not os.path.exists(CKPT_PATH):
    os.makedirs(CKPT_PATH)

# define training and validation data loaders
data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4,
    collate_fn=utils.collate_fn)

data_loader_val = torch.utils.data.DataLoader(
    dataset_val, batch_size=1, shuffle=False, num_workers=4,
    collate_fn=utils.collate_fn)

# move model to the right device
#model = torch.nn.parallel.DistributedDataParallel(model)
model.to(device)

# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=LR,
                            momentum=MOMENTUM, 
                            weight_decay=WEIGHT_DECAY)
# and a learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                            step_size=3,
                                            gamma=0.1)

disc_str = ''.join(args.disciplines)

best_ap = 0.0
for epoch in range(EPOCHS):
    # train for one epoch, printing every 10 iterations
    train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
    # update the learning rate
    lr_scheduler.step()
    # evaluate on the test dataset
    ap = evaluate(model, data_loader, device=device)

    if ap > best_ap:
        best_ap = ap
        torch.save(model.state_dict(), os.path.join(CKPT_PATH, f'{args.model}_{disc_str}_{args.split}.pth'))

print("That's it!")