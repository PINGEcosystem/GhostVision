'''

'''

import os, sys
from inference import get_model
import supervision as sv
from trackers import SORTTracker
from trackers import DeepSORTFeatureExtractor, DeepSORTTracker
import numpy as np
import time
import pandas as pd

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
    crabDetections = pd.concat(allCrabPreds)

    out_file = out_vid.replace('.mp4', '_ALL.csv')
    crabDetections.to_csv(out_file, index=False)

    return