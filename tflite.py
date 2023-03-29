import numpy as np
import cv2
import tensorflow as tf
import os
from keras.utils import np_utils


def tflite_convert(model):
    model.save("tensorflow.h5")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    open('converted.tflite', 'wb').write(tflite_model)
    return

def tflite_suiron(tflitemodel,input_data):
    # Load the TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(tflitemodel)
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    input_shape = input_details[0]['shape']
    interpreter.set_tensor(input_details[0]['index'], input_data)
    #interpreter.set_tensor(input_details[0]['index'],  x_test)

    interpreter.invoke()

    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data

