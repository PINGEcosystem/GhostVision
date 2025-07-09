
'''

'''

import os

from pingmapper.funcs_common import *
from pingmapper.class_rectObj import rectObj
# from inference_folder import do_inference
# from inference_tracker import do_tracker_inference
from ghostvisionRF.inference_folder import do_inference
from ghostvisionRF.inference_tracker import do_tracker_inference
import cv2


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
    def _detectCrabPots(self, in_dir: str, export_image: bool=True, export_vid: bool=False, confidence: float=0.5, iou_threshold: float=0.5):
        '''
        '''

        # Do inference
        do_inference(in_dir=in_dir, export_image=export_image, confidence=confidence, iou_threshold=iou_threshold)

        return
    
    #=======================================================================
    def _detectTrackCrabPots(self, in_vid: str, confidence: float=0.5, iou_threshold: float=0.5, stride: float=0.2, nchunk: int=500):
        '''
        '''

        wcp_dir_name = 'wcp_mw'

        # Get wcp folder
        wcp_dir = os.path.join(self.outDir, wcp_dir_name)

        # Do inference
        do_tracker_inference(in_vid=in_vid, confidence=confidence, iou_threshold=iou_threshold, stride=stride, nchunk=nchunk)

        return

        