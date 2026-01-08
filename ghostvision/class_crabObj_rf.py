
'''
Copyright (c) 2025 Cameron S. Bodine
'''

import os, sys

# # For debug
# from class_rectObj import rectObj

from pingmapper.class_rectObj import rectObj

import ast
import numpy as np
import pandas as pd
import geopandas as gpd

from pingdetect.rf_infer_folder import do_inference
from pingdetect.rf_infer_tracker import do_tracker_inference


class crabObj(rectObj):

    '''
    '''

    ############################################################################
    # Create crabObj() instance from previously created rectObj() instance     #
    ############################################################################

    #=======================================================================
    def __init__(self,
                 metaFile):

        rectObj.__init__(self, metaFile)

        return
    
    #=======================================================================
    def _detectCrabPots(self, rf_model: str, in_dir: str, out_dir: str, detect_csv: str, export_image: bool=True, export_vid: bool=False, confidence: float=0.5, iou_threshold: float=0.5):
        '''
        '''

        # Do inference
        do_inference(rf_model=rf_model, in_dir=in_dir, out_dir=out_dir, detect_csv=detect_csv, export_image=export_image, confidence=confidence, iou_threshold=iou_threshold)

        return
    
    #=======================================================================
    def _detectTrackCrabPots(self, rf_model: str, in_vids: list, confidence: float=0.5, iou_threshold: float=0.5, stride: float=0.2, nchunk: int=500):
        '''
        '''

        wcp_dir_name = 'wcp_mw'

        # Get wcp folder
        wcp_dir = os.path.join(self.outDir, wcp_dir_name)

        # Do inference
        do_tracker_inference(rf_model=rf_model, in_vids=in_vids, confidence=confidence, iou_threshold=iou_threshold, stride=stride, nchunk=nchunk, debug_export_frames=False)

        return
    
    