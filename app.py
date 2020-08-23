# way to upload image: endpoint
# way to save the image
# function to make prediction on the image
# show the results
import os
import torch
import cv2

import albumentations
import pretrainedmodels

import numpy as np
import torch.nn as nn

import torch
import torchvision
import matplotlib.pyplot as plt

from flask import Flask
from flask import request
from flask import render_template
from torch.nn import functional as F
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import  FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator


app = Flask(__name__)
UPLOAD_FOLDER = "api_image_store"
DEVICE = "cpu"
MODEL = None

def model():
    # load the COCO pre-trained model
    # we will keep the image size to 1024 pixels instead of the original 800,
    # this will ensure better training and testing results, although it may...
    # ... increase the training time (a tarde-off)
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True, 
                                                                 min_size=1024)
    # one class is pneumonia, and the other is background
    num_classes = 2
    # get the input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace pre-trained head with our features head
    # the head layer will classify the images based on our data input features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model

def format_prediction_string(boxes, scores):
    pred_strings = []
    for j in zip(scores, boxes):
        pred_strings.append("Confidence: {0:.4f} Coords: {1} {2} {3} {4}".format(j[0]*100, j[1][0], j[1][1], j[1][2], j[1][3]))

    return " ".join(pred_strings)

def predict(image_path, model):
    model.to(DEVICE)
    results = []
    detection_threshold = 0.9
    model.eval()
    with torch.no_grad():
        test_images = image_path
        orig_image = cv2.imread(test_images, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
        image = image/255.0
        image = np.transpose(image, (2, 0, 1)).astype(np.float)
        image = torch.tensor(image, dtype=torch.float).to(DEVICE)
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
            print('PATH.......', image_path)
            plt.savefig(f"static/prediction/{image_path.split(os.path.sep)[-1]}")
            plt.close()
                    
            result = {
                # 'patientId': test_images[i].split('.')[0],
                'Prediction': format_prediction_string(boxes, scores)
            }
            results.append(result)
        else:
            result = {
                # 'patientId': test_images[i].split('.')[0],
                'Prediction': None
            }
            results.append(result)

    if results[0]['Prediction'] == None or results[0]['Prediction'] == '':
        # orig_image = cv2.imread(test_images, cv2.IMREAD_COLOR)
        # plt.imshow(cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB))
        # plt.axis('off')
        # plt.savefig(f"static/prediction/{image_path.split(os.path.sep)[-1]}")
        # plt.close()
        return 'No Pneumonia found. Patient is Healthy.'
    else:
        return results[0]


@app.route("/", methods=["GET", "POST"])
def upload_predict():
    if request.method == "POST":
        image_file = request.files["image"]
        if image_file:
            image_location = os.path.join(
                UPLOAD_FOLDER,
                image_file.filename
            )
            image_file.save(image_location)
            pred = predict(image_location, MODEL)
            return render_template("index.html", prediction=pred, image_loc=True, name=image_file.filename)
    return render_template("index.html", prediction=0, image_loc=None)


if __name__ == "__main__":
    MODEL = model()
    MODEL.load_state_dict(torch.load("fasterrcnn_resnet50_fpn.pth", map_location=torch.device(DEVICE)))
    MODEL.to(DEVICE)
    app.run(host="127.0.0.1", port=12000, debug=True)