from ctypes import _NamedFuncPointer
import cv2
import numpy as np
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image

# Load the pre-trained InceptionV3 model
model = InceptionV3(weights='imagenet')

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(299, 299))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

def predict_skin_disease(img_path):
    img_array = preprocess_image(img_path)
    predictions = model.predict(img_array)
    decoded_predictions = decode_predictions(predictions)[0]
    top_prediction = decoded_predictions[0]
    return top_prediction

def display_result(img_path, result):
    img = cv2.imread(img_path)
    cv2.imshow("Skin Disease Image", img)
    print(f"Prediction: {result[1]}, Confidence: {result[2]*100:.2f}%")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if _NamedFuncPointer == "_main_":
    # Replace 'your_image.jpg' with the path to your skin disease image
    image_path = 'skin.jpg'
    prediction_result = predict_skin_disease(image_path)
    display_result(image_path, prediction_result)