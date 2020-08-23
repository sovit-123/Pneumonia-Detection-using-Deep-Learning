import pandas as pd
import numpy as np
import cv2
import os
import re
import albumentations as A
import torch
import torchvision

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SequentialSampler
from PIL import Image
from albumentations.pytorch.transforms import ToTensorV2
from matplotlib import pyplot as plt
from tqdm import tqdm

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

DIR_INPUT = '../input'
DIR_TEST = f"{DIR_INPUT}/samples"
test_images = os.listdir(DIR_TEST)
print(f"Test instances: {len(test_images)}")

# load COCO pretrained model
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
num_classes = 2  # 1 class is pneumonia and the other is background
# get the number of input features for the classifier
in_features = model.roi_heads.box_predictor.cls_score.in_features
# fine-tune head
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

os.makedirs('../test_predictions', exist_ok=True)
model.load_state_dict(torch.load('fasterrcnn_resnet50_fpn.pth'))
model.to(device)

def format_prediction_string(boxes, scores):
    pred_strings = []
    for j in zip(scores, boxes):
        pred_strings.append("{0:.4f} {1} {2} {3} {4}".format(j[0], j[1][0], j[1][1], j[1][2], j[1][3]))

    return " ".join(pred_strings)

detection_threshold = 0.9
img_num = 0
results = []
model.eval()
with torch.no_grad():
    for i, image in tqdm(enumerate(test_images), total=len(test_images)):

        orig_image = cv2.imread(f"{DIR_TEST}/{test_images[i]}", cv2.IMREAD_COLOR)
        image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
        image = image/255.0
        image = np.transpose(image, (2, 0, 1)).astype(np.float)
        image = torch.tensor(image, dtype=torch.float).cuda()
        image = torch.unsqueeze(image, 0)

        cpu_device = torch.device("cpu")

        outputs = model(image)
        
        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        if len(outputs[0]['boxes']) != 0:
            for counter in range(len(outputs[0]['boxes'])):
                boxes = outputs[0]['boxes'].data.cpu().numpy()
                scores = outputs[0]['scores'].data.cpu().numpy()
                boxes = boxes[scores >= detection_threshold].astype(np.int32)
                draw_boxes = boxes.copy()
                boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
                boxes[:, 3] = boxes[:, 3] - boxes[:, 1]
            
            for box in draw_boxes:
                cv2.rectangle(orig_image,
                            (int(box[0]), int(box[1])),
                            (int(box[2]), int(box[3])),
                            (0, 0, 255), 3)
        
            plt.imshow(cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.savefig(f"../test_predictions/{test_images[i]}")
            plt.close()
                
            result = {
                'patientId': test_images[i].split('.')[0],
                'PredictionString': format_prediction_string(boxes, scores)
            }
            results.append(result)
        else:
            result = {
                'patientId': test_images[i].split('.')[0],
                'PredictionString': None
            }
            results.append(result)

sub_df = pd.DataFrame(results, columns=['patientId', 'PredictionString'])
sub_df.head()
sub_df.to_csv('submission.csv', index=False)