
'''

'''

import os

from pingmapper.funcs_common import *
from pingmapper.class_rectObj import rectObj
from inference_folder import do_inference
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
    def _detectCrabPots(self, export_image=True, export_vid=False, confidence=0.5, iou_threshold=0.5):
        '''
        '''

        wcp_dir_name = 'wcp_mw'

        # Get wcp folder
        wcp_dir = os.path.join(self.outDir, wcp_dir_name)

        # Do inference
        do_inference(in_dir=wcp_dir, export_image=export_image, confidence=confidence, iou_threshold=iou_threshold)

        