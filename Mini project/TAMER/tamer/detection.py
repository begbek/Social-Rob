import keras
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np


face_classifier = cv2.CascadeClassifier5('/haarcascade_frontalface_default.xml')
classifier = load_model('/EmotionDetectionModel.h5')