from math import ceil, floor
from tkinter import Y
from flask import Flask, render_template, request
import os
from platform import python_branch
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import cv2
from glob import glob
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.utils import CustomObjectScope
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score
from metrics import dice_loss, dice_coef, iou

H = 512
W = 512

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def read_image(path):
    x = cv2.imread(path, cv2.IMREAD_COLOR)
    # x = cv2.resize(x, (W, H))
    ori_x = x
    x = x/255.0
    x = x.astype(np.float32)
    return ori_x, x

def read_mask(path):
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  ## (512, 512)
    # x = cv2.resize(x, (W, H))
    ori_x = x
    x = x/255.0
    x = x.astype(np.int32)
    return ori_x, x

def load_data(path):
    x = sorted(glob(os.path.join(path, "image", "*.jpg")))
    y = sorted(glob(os.path.join(path, "mask", "*.jpg")))
    return x, y

def save_results(ori_x, ori_y, y_pred, save_image_path):
    line = np.ones((512, 10, 3)) * 255

    ori_y = np.expand_dims(ori_y, axis=-1)
    ori_y = np.concatenate([ori_y, ori_y, ori_y], axis=-1)

    # plt.imshow(y_pred, interpolation='nearest')
    # plt.show()
    y_pred = np.expand_dims(y_pred, axis=-1) 
    # plt.imshow(y_pred, interpolation='nearest')
    # plt.show()
    y_pred = np.concatenate([y_pred, y_pred, y_pred], axis=-1) 
    # plt.imshow(y_pred, interpolation='nearest')
    # plt.show()

    cat_images = np.concatenate([ori_x, line, ori_y,line,y_pred], axis=1)
    cv2.imwrite(save_image_path, cat_images)

app = Flask(__name__)
@app.route("/", methods=['GET', 'POST'])
def main():
	return render_template("index.html")

@app.route("/about")
def about_page():
	return "Image Segmentation"

@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
    if request.method == 'POST':
        img = request.files['image']

        img_path = "saveimages/test/image/" + img.filename	
        img.save(img_path)
		
        maskimg = request.files['mask']

        mask_img_path = "saveimages/test/mask/" + maskimg.filename	
        maskimg.save(mask_img_path)

        save_image_path = ""
        create_dir("flask_results")
        """ Load the model """
        with CustomObjectScope({'iou': iou, 'dice_coef': dice_coef, 'dice_loss': dice_loss}):
            model = tf.keras.models.load_model("files\\model.h5")

        """ Load the dataset """
        # dataset_path = os.path.join("saveimages","test")
        # print(dataset_path)
        test_x, test_y = img_path,mask_img_path
        
        print(len(test_x), len(test_y))
        """ Make the prediction and calculate the metrics values """
        SCORE = []
        print(11)
        if 1 == 1: 
            x, y = test_x, test_y
            """ Extracting name """
            name = x.split("/")[-1].split(".")[0]
            
            print(x,y)
            """ Read the image and mask """
            ori_x, x = read_image(x)
            

            """ Prediction """
            print(12)
            y_pred = model.predict(np.expand_dims(x, axis=0))[0]
            
            # print(y_pred)
            
            y_pred = y_pred > 0.5
            y_pred = y_pred.astype(np.int32)
            y_pred = np.squeeze(y_pred, axis=-1)
            print(y_pred)
            print("success")
            print(y_pred * 255)
            # plt.imshow(y_pred, interpolation='nearest')
            # plt.show()

            # print(y_pred)
            
            """ Saving the images """
            print(y_pred)
            save_image_path = f"static\\{name}.png"
            ans = y_pred * 255
            print('2222')
            print(ans)
            print('11111')
            ori_y, y = read_mask(y)
            # cv2.imwrite(save_image_path, ans)
            y_pred = np.expand_dims(y_pred, axis=-1) 
            save_results(ori_x, ans*255,ori_y, save_image_path)
            print("saved")
            print(name)
            
            """ Flatten the array """
            y = y.flatten()
            y_pred = y_pred.flatten()

            # """ Calculate the metrics """
            acc_value =  str(accuracy_score(y, y_pred))[:6]
            # f1_value = str(f1_score(y, y_pred, labels=[0, 1], average="binary"))[:6]
            # jac_value = str(jaccard_score(y, y_pred, labels=[0, 1], average="binary"))[:6]
            recall_value = str(recall_score(y, y_pred, labels=[0, 1], average="binary"))[:6]
            precision_value = str(precision_score(y, y_pred, labels=[0, 1], average="binary"))[:6]
            
            # SCORE.append([name, acc_value, f1_value, jac_value, recall_value, precision_value])
        # print(13)
        # score = [s[1:] for s in SCORE]
        # score = np.mean(score, axis=0)
        # p = 1
        # print(f"Accuracy: {score[0]:0.5f}")
        # print(f"F1: {score[1]:0.5f}")
        # print(f"Jaccard: {score[2]:0.5f}")
        # print(f"Recall: {score[3]:0.5f}")
        # print(f"Precision: {score[4]:0.5f}")

        """ Saving """
        # df = pd.DataFrame(SCORE, columns=["Image", "Acc", "F1", "Jaccard", "Recall", "Precision"])
        # df.to_csv("files/score.csv")
    return render_template("index.html", acc_value = acc_value,recall_value= recall_value,precision_value = precision_value, img_path = save_image_path)

if __name__ =='__main__':
	#app.debug = True
	app.run(debug = True)