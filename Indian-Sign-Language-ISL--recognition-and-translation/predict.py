"""
Contains functions : pre_process() and which() that are needed by translator.py for predicting image from webcam
"""
import numpy as np
import cv2
from keras.models import load_model
import os

print("Everything is installed and working!")

from variables import *
from keras.models import load_model


# Loads pretrained CNN Model from MODEL_PATH
try:
    model = load_model(MODEL_PATH)
    print(f"Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    print(f"Error loading model: {e}")
    raise


def pre_process(img_array):
    """
    :param img_array: image converted to np array
    :return:  img_array after pre-processing (grayscale, resize, normalize)
    """
    try:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
        img_array = cv2.resize(img_array, (IMAGE_SIZE, IMAGE_SIZE))
        img_array = img_array.reshape(IMAGE_SIZE, IMAGE_SIZE, 1)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        print(f"Error in pre-processing: {e}")
        raise



def which(img_array):
    """
    :param img_array: np array of image which is to be predicted
    :return: confidence precentage and predicted letter
    """
    try:
        img_array = pre_process(img_array)
        preds = model.predict(img_array, verbose=0)  # Reduce verbosity
        preds *= 100
        most_likely_class_index = int(np.argmax(preds))
        confidence = preds.max()
        
        # Only return prediction if confidence is above threshold
        if confidence >= THRESHOLD:
            return confidence, LABELS[most_likely_class_index]
        else:
            return confidence, "Unknown"
    except Exception as e:
        print(f"Error in prediction: {e}")
        return 0, "Error"
