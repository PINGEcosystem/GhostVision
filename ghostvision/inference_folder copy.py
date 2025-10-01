
'''
Copyright (c) 2025 Cameron S. Bodine
'''

import sys, os
import onnxruntime as ort
import numpy as np
from PIL import Image
from ultralytics import YOLO
import json
import torch

from inference.models.yolov10 import YOLOv10ObjectDetection

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def get_model(dest_dir: str):


    # Determine the model type
    with open(os.path.join(dest_dir, 'model_type.json'), 'r') as f:
        data = json.load(f)
        model_type = data['model_type']

        pred_task = data['project_task_type']
        if pred_task == 'object-detection':
            pred_task = 'detect'

    # Convert to yolo
    if 'yolo' in model_type:
        model = YOLO(os.path.join(dest_dir, 'weights.onnx'), task=pred_task)

    return model

def preprocess_image(image_path, image_size=(224, 224)):
    img = Image.open(image_path)

    # Resize the image
    img = img.resize(image_size)

    # Convert the image to a NumPy array and normalize it
    img_np = np.array(img).astype(np.float32) / 255.0

    # Transpose the image to have channels first (if needed)
    img_np = img_np.transpose((2, 0, 1))

    # Add a batch dimension
    img_np = np.expand_dims(img_np, axis=0)
    return img_np


def do_inference(dir: str):
    '''
    Run inference on input folder
    '''

    print(dir)
    print(os.path.exists(dir))

    # Load the model
    model_path = r'Z:\UDEL\PythonRepos\GhostVision\ghostvision\allcrabpotsources\11\weights.onnx'

    ort_sess = ort.InferenceSession(model_path)

    # Get the names of the outputs
    output_names = [output.name for output in ort_sess.get_outputs()]

    # Get input metadata
    input_metadata = ort_sess.get_inputs()
    for input_tensor in input_metadata:
        print(f"Input name: {input_tensor.name}")
        print(f"Input shape: {input_tensor.shape}")
        print(f"Input type: {input_tensor.type}")

    # Get input shape for ort_sess
    input_shape = input_metadata[0].shape


    # Get images in directory
    images = os.listdir(dir)
    images = [os.path.join(dir, img) for img in images if img.endswith('.jpg') or img.endswith('.png')]

    # model = torch.hub.load('.', 'custom', path=model_path, force_reload=True)

    for img in images:
        print(f"\n\nProcessing {img}")

        
        input_name = ort_sess.get_inputs()[0].name
        output_name = ort_sess.get_outputs()[0].name
        # print(ort_sess.get_outputs().name)
        for v in ort_sess.get_outputs():
            print(v.name)
        preprocessed_image = preprocess_image(img, image_size=(input_shape[2], input_shape[3]))
        inname = [i.name for i in ort_sess.get_inputs()]
        inp = {inname[0]: preprocessed_image}
        results = ort_sess.run([output_name], inp)[0]

        # out_filename = f'./test.jpg'
        # open(out_filename, 'wb').write(results)
        # from PIL import Image
        # Image.open(out_filename).show()

        # print(vars(results))
        # sys.exit()

        print(f"Output names: {output_names}")

        # Assuming the model outputs bounding boxes and scores
        bounding_boxes = results[0]  # Replace with the correct index for bounding boxes
        # scores = results[1]          # Replace with the correct index for scores

        # Optionally, filter results based on a confidence threshold
        confidence_threshold = 0.5
        for box in bounding_boxes:
            print(f"Box: {box}")
            print(len(box))
            sys.exit()

        



    
    

    return