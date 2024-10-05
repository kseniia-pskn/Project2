import os
import cv2
import numpy as np
import tensorflow as tf

keras = tf.keras
load_model = keras.models.load_model
Model = keras.models.Model

"""
This is a Singleton class which bears the ml model in memory
"""
BASE = os.path.dirname(os.path.abspath(__file__))

class HandShapeFeatureExtractor:
    __single = None

    @staticmethod
    def get_instance():
        if HandShapeFeatureExtractor.__single is None:
            HandShapeFeatureExtractor()
        return HandShapeFeatureExtractor.__single

    def __init__(self):
        if HandShapeFeatureExtractor.__single is None:
            try:
                # Load the pre-trained model
                model_path = os.path.join(BASE, 'gestures_trained_cnn_model.h5')
                if not os.path.exists(model_path):
                    raise FileNotFoundError(f"Model file not found at {model_path}")
                real_model = load_model(model_path)
                
                self.model = real_model
                HandShapeFeatureExtractor.__single = self
            except Exception as e:
                print(f"Error loading the model: {str(e)}")
                raise
        else:
            raise Exception("This class bears the model, so it is made Singleton")

    # private method to preprocess the image
    @staticmethod
    def __pre_process_input_image(crop):
        try:
            if not isinstance(crop, np.ndarray):
                raise TypeError("Input to __pre_process_input_image must be a numpy array.")

            img = cv2.resize(crop, (300, 300))
            img_arr = np.array(img) / 255.0
            img_arr = np.stack((img_arr,) * 3, axis=-1)  # Ensure the image has 3 channels (RGB)
            img_arr = img_arr.reshape(1, 300, 300, 3)  # Reshape to match model input shape
            return img_arr
        except Exception as e:
            print(f"Error during preprocessing the image: {str(e)}")
            raise

    def extract_feature(self, image):
        try:
            if not isinstance(image, np.ndarray):
                raise TypeError("Input to extract_feature must be a numpy array.")

            img_arr = self.__pre_process_input_image(image)
            feature_vector = self.model.predict(img_arr)

            # Ensure the output is a 1-dimensional vector
            if len(feature_vector.shape) > 1:
                feature_vector = feature_vector.flatten()
            
            return feature_vector
        except Exception as e:
            print(f"Error during feature extraction: {str(e)}")
            raise
