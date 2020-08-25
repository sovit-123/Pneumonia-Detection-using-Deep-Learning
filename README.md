# Pneumonia Detection using Deep Learning (PyTorch)



**[Try the live demo here](https://dashboard.heroku.com/apps/pneumonia-detection-dl) => https://dashboard.heroku.com/apps/pneumonia-detection-dl**



## <u>About the Project</u>

***This project aims to detect pneumonia from chest radio graph images. This project uses data from the [RSNA Pneumonia Detection Challenge](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/overview). The results and evaluation metric plots presented here also uses the same metric as was in the competition.*** 

![](https://github.com/sovit-123/Pneumonia-Detection-using-Deep-Learning/blob/master/preview_image/preview_image.jpg?raw=true)



***Feel free to improve upon the project by using the [TRAINING Kaggle notebook](https://www.kaggle.com/sovitrath/rsna-pytorch-hackathon-fasterrcnn-resnet-training/notebook) and [TEST Kaggle notebook](https://www.kaggle.com/sovitrath/rsna-pytorch-hackathon-fasterrcnn-resnet-test/notebook).***



## <u>Framework Used</u>

***This project used the PyTorch deep learning framework. Use PyTorch version >= 1.4 to reproduce the results.***



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



## <u>Run the Detection API on Your LocalHost</u>

* **I have made this really simple.**

![](https://github.com/sovit-123/Pneumonia-Detection-using-Deep-Learning/blob/master/some_results/api_new.PNG?raw=true)

1. Clone this repository to your local disk.
2. Run the `requirements.txt` file using `pip install -r requirements.txt`.
3. Run the `app.py` script using `python app.py`.
4. Open `127.0.0.1:12000` in your browser. 
5. Choose a lung x-ray image from your disk and click on the `Predict` button. 



## <u>Training and Testing</u>

* **First run** `dcm_to_jpg.py` to convert all the DICOM images to JPG images and save them in the `inout/images` folder. Change  the paths according to your need if want to structure your project differently.
* **Then run**  `train.py` to train a **Faster RCNN ResNet50** backbone model on the data. I have trained the model for 30 epochs to obtain the results. You may train for longer if you want to.
* **Finally run** `test.py` to predict on the test images present in the `input/samples` folder.

***If You Want to Directly Predict on the Test Images, Then Download the Weights from [THIS KAGGLE NOTEBOOK](https://www.kaggle.com/sovitrath/rsna-pytorch-hackathon-fasterrcnn-resnet-training/output).***

***You will also find the [test notebook here](https://www.kaggle.com/sovitrath/rsna-pytorch-hackathon-fasterrcnn-resnet-test/notebook).***



## <u>Results</u>

* The following the private and public leaderboard score as per the competition metric of Average Precision.

  |                       | Private Leaderboard | Public  Leaderboard |
  | --------------------- | ------------------- | ------------------- |
  | **Average Precision** | 0.12993             | 0.11904             |

* **Validation Precision Plot**

  ![](https://github.com/sovit-123/Pneumonia-Detection-using-Deep-Learning/blob/master/precision.png?raw=true)

* **Loss Plot**

  ![](https://github.com/sovit-123/Pneumonia-Detection-using-Deep-Learning/blob/master/loss.png?raw=true)

* **Detection on a Test Image**

  ![](https://github.com/sovit-123/Pneumonia-Detection-using-Deep-Learning/blob/master/some_results/00ad18b7-06ee-4c4d-abca-14bdf814e8b2.jpg?raw=true)

## <u>References</u>

* [Deep Learning for Automatic Pneumonia Detection](https://arxiv.org/pdf/2005.13899v1.pdf).
* [Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks.](https://arxiv.org/abs/1506.01497)