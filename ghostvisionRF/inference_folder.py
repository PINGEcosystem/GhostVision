
'''
Copyright (c) 2025 Cameron S. Bodine
'''

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow INFO and WARNING messages

import warnings
warnings.filterwarnings("ignore")

import sys
from inference import get_model
import supervision as sv
import json
import cv2
import pandas as pd
import shutil
import time
from tqdm import tqdm
from joblib import Parallel, delayed, cpu_count

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# def get_model(dest_dir: str):


#     # Determine the model type
#     with open(os.path.join(dest_dir, 'model_type.json'), 'r') as f:
#         data = json.load(f)
#         model_type = data['model_type']

#         pred_task = data['project_task_type']
#         if pred_task == 'object-detection':
#             pred_task = 'detect'

#     # Convert to yolo
#     if 'yolo' in model_type:
#         model = YOLO(os.path.join(dest_dir, 'weights.onnx'), task=pred_task)

#     return model

# def preprocess_image(image_path, image_size=(224, 224)):
#     img = Image.open(image_path)

#     # Resize the image
#     img = img.resize(image_size)

#     # Convert the image to a NumPy array and normalize it
#     img_np = np.array(img).astype(np.float32) / 255.0

#     # Transpose the image to have channels first (if needed)
#     img_np = img_np.transpose((2, 0, 1))

#     # Add a batch dimension
#     img_np = np.expand_dims(img_np, axis=0)
#     return img_np


def do_inference(in_dir: str, out_dir: str, detect_csv: str, export_image: bool=True, export_vid: bool=False, confidence: float=0.2, iou_threshold: float=0.2):
    '''
    Run inference on input folder
    '''

    print(in_dir)
    print(os.path.exists(in_dir))
    print('confidence: {}\tiou: {}'.format(confidence, iou_threshold))

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    else:
        shutil.rmtree(out_dir)
        os.makedirs(out_dir)

    # ort_sess = ort.InferenceSession(model_path)
    model = get_model('allcrabpotsources/11')

    # Get images in directory
    images = os.listdir(in_dir)
    images = [os.path.join(in_dir, img) for img in images if img.endswith('.jpg') or img.endswith('.png')]

    # Batch inference
    start_time = time.time()
    results = model.infer(images, confidence=confidence, iou_threshold=iou_threshold, show_progress=True)
    print("\n\nInference Time (s):", round(time.time() - start_time, ndigits=1))

    # Save results and export images
    start_time = time.time()
    # save_results_image(out_dir=out_dir, detect_csv=detect_csv, results=results, images=images, export_image=export_image)
    r = Parallel(n_jobs=cpu_count())(delayed(save_results_image)(out_dir=out_dir, result=res, image=img, export_image=export_image) for res, img in tqdm(zip(results, images), total=len(results)))

    dfAll = pd.concat(r, axis=0)

    # Drop nan items
    dfAll = dfAll.dropna(subset=['class_name'])

    dfAll.to_csv(detect_csv, index=False)

    print("\n\nSave Results Time (s):", round(time.time() - start_time, ndigits=1))



def save_results_image(out_dir: str, result: list, image: str, export_image: bool=True):
    '''
    
    '''
        
    # load the results into the supervision Detections api
    detections = sv.Detections.from_inference(result)
    
    # Prepare label for annotations
    labels = [f"{class_id['class_name']} {confidence:0.2f}" for _, _, confidence, _, _, class_id in detections]

    # create supervision annotators
    bounding_box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    # Open the image
    img_nd = cv2.imread(image)

    # annotate the image with our inference results
    annotated_image = bounding_box_annotator.annotate(
                    scene=img_nd, detections=detections)
    annotated_image = label_annotator.annotate(
        scene=annotated_image, detections=detections, labels=labels)

    in_file = os.path.basename(image)
    file_name = in_file.replace('.png', '_detect.png')
    file_name = os.path.join(out_dir, file_name)
    
    if export_image:
        cv2.imwrite(file_name, annotated_image)

    result = result.json()
    result = json.loads(result)

    # Prepare dataframe
    # df = pd.DataFrame.from_dict({'chunk':[i], 'beam':[self.beamName], 'name':[os.path.basename(file_name)]})
    df = pd.DataFrame.from_dict({'name':[os.path.basename(file_name)]})
    df1 = pd.json_normalize(result['image'])
    df1 = df1.rename(columns={'width': 'img_width', 'height': 'img_height'})
    df2 = pd.json_normalize(result['predictions'])

    df = pd.concat([df, df1, df2], axis=1)

    # If multiple predictions in an image
    if len(df) > 1:
        # df['chunk'] = i
        df['name'] = os.path.basename(file_name)
        df['img_width'] = df.loc[0, 'img_width']
        df['img_height'] = df.loc[0, 'img_height']

    df['chunk_id'] = int(file_name.split('.jpg')[0].split('_')[-2])
    df['chunk_offset'] = int(file_name.split('.jpg')[0].split('_')[-1])

    df = df.rename(columns={'name': 'name_long'})

    return df
