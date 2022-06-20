from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing import image
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import pandas as pd
import cv2
from glob import glob
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.utils import CustomObjectScope
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score
from metrics import dice_loss, dice_coef, iou

app = Flask(__name__)

# dic = {0 : 'Cat', 1 : 'Dog'}

with CustomObjectScope({'iou': iou, 'dice_coef': dice_coef, 'dice_loss': dice_loss}):
	model = tf.keras.models.load_model("files\\model.h5")


def save_results( y_pred, save_image_path):
    # line = np.ones((512, 10, 3)) * 255

    # ori_y = np.expand_dims(ori_y, axis=-1)
    # ori_y = np.concatenate([ori_y, ori_y, ori_y], axis=-1)

    # y_pred = np.expand_dims(y_pred, axis=-1)
    # y_pred = np.concatenate([y_pred, y_pred, y_pred], axis=-1) * 255

    # cat_images = np.concatenate([ori_x, line, ori_y, line, y_pred], axis=1)
    cv2.imwrite(save_image_path, y_pred)

def read_image(path):
	print(cv2.imread(path, cv2.IMREAD_COLOR))
	x = cv2.imread(path, cv2.IMREAD_COLOR) 
	# # x = cv2.resize(x, (W, H))
	ori_x = x
	
	x = x / 255.0
	x = x.astype(np.float32)
	return ori_x, x

def read_mask(path):
    print(cv2.imread(path, cv2.IMREAD_GRAYSCALE))
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  ## (512, 512)
    # x = cv2.resize(x, (W, H))
    print("x" + path)
    print(type(x))
    print("y")
    ori_x = x
    x = x/255.0
    x = x.astype(np.int32)
    return ori_x, x


def pre(x,y):
	print(x,y)
	""" Read the image and mask """

	orig_x,x  = read_image(x) 
	# y = read_mask(y)
	print("abcd")
	print(type(x))
	print("abcd")
	print(model.predict(np.expand_dims(x, axis=0))[0])
	""" Prediction """
	y_pred = model.predict(np.expand_dims(x, axis=0))[0]
	y_pred = y_pred > 0.5
	y_pred = y_pred.astype(np.int32)
	y_pred = np.squeeze(y_pred, axis=-1)
	save_image_path = f"results\\ans.png"
	save_results( y_pred, save_image_path)

	y = y.flatten()
	y_pred = y_pred.flatten()
	acc_val = accuracy_score(y, y_pred)

	return acc_val
# model.make_predict_function()

# def predict_label(img_path):
# 	i = image.load_img(img_path, target_size=(100,100))
# 	i = image.img_to_array(i)/255.0
# 	i = i.reshape(1, 100,100,3)
# 	p = model.predict_classes(i)
# 	return dic[p[0]]


# routes
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

		img_path = "saveimages/" + img.filename	
		img.save(img_path)
		
		maskimg = request.files['image']

		mask_img_path = "files/" + maskimg.filename	
		img.save(mask_img_path)


		p = pre(img_path,mask_img_path)
		# p = predict_label(img_path)

	# return render_template("index.html", prediction = p, img_path = img_path)


if __name__ =='__main__':
	#app.debug = True
	app.run(debug = True)