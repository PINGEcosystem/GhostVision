'''
Copyright (c) 2025 Cameron S. Bodine
'''

import os, sys
from inference import get_model
import importlib
import supervision as sv
import numpy as np
import time
import pandas as pd

# ensure supervision.detection.utils has box_iou_batch (compatibility shim)
try:
    utils_mod = importlib.import_module('supervision.detection.utils')
except Exception:
    utils_mod = None

if utils_mod is not None and not hasattr(utils_mod, 'box_iou_batch'):
    def box_iou_batch(boxes1, boxes2):
        """
        Simple numpy implementation returning pairwise IoU matrix between boxes1 (N,4) and boxes2 (M,4).
        Boxes are [x1, y1, x2, y2].
        """
        boxes1 = np.asarray(boxes1)
        boxes2 = np.asarray(boxes2)
        if boxes1.size == 0 or boxes2.size == 0:
            return np.zeros((boxes1.shape[0], boxes2.shape[0]))
        x1 = np.maximum(boxes1[:, None, 0], boxes2[None, :, 0])
        y1 = np.maximum(boxes1[:, None, 1], boxes2[None, :, 1])
        x2 = np.minimum(boxes1[:, None, 2], boxes2[None, :, 2])
        y2 = np.minimum(boxes1[:, None, 3], boxes2[None, :, 3])
        inter_w = np.maximum(0.0, x2 - x1)
        inter_h = np.maximum(0.0, y2 - y1)
        inter = inter_w * inter_h
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        union = area1[:, None] + area2[None, :] - inter
        iou = inter / (union + 1e-9)
        return iou
    setattr(utils_mod, 'box_iou_batch', box_iou_batch)

# import trackers after shim
from trackers import SORTTracker
from trackers import DeepSORTFeatureExtractor, DeepSORTTracker

# Add at the top of your file
last_boxes = None
last_ids = None


def do_tracker_inference(in_vid: str, export_vid: bool=True, confidence: float=0.2, iou_threshold: float=0.2, stride: float=0.2, nchunk: int=500, track_prop: float=0.8):

    '''
    '''

    # Store all annotations
    allCrabPreds = []

    # Callback: process each video frame, do inference, get metadata, return annotated frame
    def callback(frame: np.ndarray, index: int) -> np.ndarray:
        global last_boxes, last_ids
        result = model.infer(frame, confidence=confidence, iou_threshold=iou_threshold)[0]

        # create supervision annotators
        bounding_box_annotator = sv.BoxAnnotator()
        label_annotator = sv.LabelAnnotator()

        # load the results into the supervision Detections api
        detections = sv.Detections.from_inference(result).with_nms(threshold=iou_threshold, class_agnostic=True)
        detections = tracker.update(detections, frame=frame)

        # print('\n\n', detections)
        
        # Prepare label for annotations
        labels = [f"{tracker_id} {confidence:0.2f}" for tracker_id, confidence in zip(detections.tracker_id, detections.confidence)]    
        # labels = [f"#{tracker_id}" for tracker_id in detections.tracker_id]
        # labels = [f"{class_id['class_name']} {confidence:0.2f}" for _, _, confidence, _, _, class_id in detections]

        # annotate the image with our inference results
        annotated_image = bounding_box_annotator.annotate(
                        scene=frame, detections=detections)
        annotated_image = label_annotator.annotate(
            scene=annotated_image, detections=detections, labels=labels)
        
        # Save current boxes and IDs for next frame
        last_boxes = detections.xyxy.copy()
        last_ids = detections.tracker_id.copy() # type: ignore

        # print(index, detections.tracker_id)

        if detections.tracker_id.size>0:
            # allCrabPreds.append(detections)

            # Build DataFrame from Detections attributes
            df = pd.DataFrame({
                'vid_frame_id':index,
                'tracker_id': detections.tracker_id.tolist(),
                'class_id': detections.class_id.tolist(),
                'data': detections.data['class_name'],
                'xyxy': detections.xyxy.tolist(),
                'confidence': detections.confidence.tolist(),
            })

            df['vid_frame_id'] = index
            df['frame_width'] = frame.shape[1]
            df['frame_height'] = frame.shape[0]

            allCrabPreds.append(df)            

        return annotated_image

    # Get the model, tracker, and annotator
    model = get_model('allcrabpotsources/11')

    # minimum_consecutive_frames = int((nchunk / (nchunk*stride)) * track_prop)
    # print("Minimum Consecutive Frames: {}".format(minimum_consecutive_frames))

    feature_extractor = DeepSORTFeatureExtractor.from_timm(model_name="mobilenetv4_conv_small.e1200_r224_in1k")
    tracker = DeepSORTTracker(feature_extractor=feature_extractor,
                              lost_track_buffer=10,
                              frame_rate=10,
                              track_activation_threshold=0.1,                              
                              minimum_consecutive_frames=1,
                              minimum_iou_threshold=iou_threshold,
                              appearance_threshold=0.8,
                              appearance_weight=0.5,
                              distance_metric='cos',
                              )

    tracker.reset()
    annotator = sv.LabelAnnotator(text_position=sv.Position.TOP_CENTER)

    # Prep output name
    out_dir = os.path.dirname(in_vid)
    in_vid_name = os.path.basename(in_vid)
    out_vid_name = in_vid_name.replace('.mp4', '_track.mp4')
    out_vid = os.path.join(out_dir, out_vid_name)

    # Do inference
    start_time = time.time()
    sv.process_video(source_path=in_vid, target_path=out_vid, callback=callback, show_progress=True)
    print("\n\nInference Time (s):", round(time.time() - start_time, ndigits=1))
    
    # Extract detections
    if len(allCrabPreds) == 0:
        return
    else:
        crabDetections = pd.concat(allCrabPreds)

        out_file = out_vid.replace('.mp4', '_ALL.csv')
        crabDetections.to_csv(out_file, index=False)

        return