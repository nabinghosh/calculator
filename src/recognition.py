import tensorflow as tf
import onnxruntime as ort
import numpy as np

onnx_model_path = "models/math_cnn_model.onnx"
tflite_model_path = "models/math_cnn_model.tflite"

onnx_session = ort.InferenceSession(onnx_model_path)

interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()


input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def recognize_expression(image):
    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    expression = process_output(output_data)
    return expression

def process_output(output_data):
    pass