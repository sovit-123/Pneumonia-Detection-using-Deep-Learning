# Pneumonia Detection using Deep Learning



## <u>About the Project</u>

***This project aims to detect pneumonia from chest radio graph images. This project uses data from the [RSNA Pneumonia Detection Challenge](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/overview). The results and evaluation metric plots presented here also uses the same metric as was in the competition.*** 



***Feel free to improve upon the project by using the [TRAINING Kaggle notebook](https://www.kaggle.com/sovitrath/rsna-pytorch-hackathon-fasterrcnn-resnet-training/notebook) and [TEST Kaggle notebook](https://www.kaggle.com/sovitrath/rsna-pytorch-hackathon-fasterrcnn-resnet-test/notebook).***



## <u>Get the Data</u>

**[Get the competition data from here.](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/overview)**



## <u>Project Structure</u>

The following is the directory structure of the project. You will be able to directly 

```
───faster_rcnn_resnet_final_project
│   │   dataset.py
│   │   dcm_to_jpg.py
│   │   engine..py
│   │   fasterrcnn_resnet50_fpn.pth
│   │   loss.png
│   │   model.py
│   │   precision.png
│   │   README.md
│   │   submission.csv
│   │   test.py
│   │   train.py
├───input
│   │   stage_2_detailed_class_info.csv
│   │   stage_2_sample_submission.csv
│   │   stage_2_train_labels.csv
│   ├───images
│   ├───samples
├───test_predictions
```

* `faster_rcnn_resnet_final_project`: This contains all the files including the python scripts that you see in this repository.
* `input` : This contains the training images in JPG format in the `images` folder and test images in the `samples` folder.
* `test_predictions` will contain all the predicted bounding box results on the test images after you run `test.py`



## <u>Training and Testing</u>

* **First run** `dcm_to_jpg.py` to convert all the DICOM images to JPG images and save them in the `inout/images` folder. Change  the paths according to your need if want to structure your project differently.
* **Then run**  `train.py` to train a **Faster RCNN ResNet50** backbone model on the data. I have trained the model for 30 epochs to obtain the results. You may train for longer if you want to.
* **Finally run** `test.py` to predict on the test images present in the `input/samples` folder.

***If You Want to Directly Predict on the Test Images, Then Download the Weights from [THIS KAGGLE NOTEBOOK](https://www.kaggle.com/sovitrath/rsna-pytorch-hackathon-fasterrcnn-resnet-training/output).***

***You will also find the [test notebook here](https://www.kaggle.com/sovitrath/rsna-pytorch-hackathon-fasterrcnn-resnet-test/notebook).***







