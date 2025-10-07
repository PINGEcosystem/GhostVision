
'''

'''

import os, sys
import cv2
from ghostvision.class_crabObj_rf import crabObj

from joblib import Parallel, delayed, cpu_count
import pandas as pd
import geopandas as gpd
from glob import glob

# # Debug
# detectPath = os.path.normpath('../PINGDetect')
# detectPath = os.path.abspath(detectPath)
# sys.path.insert(0, detectPath)

from pingdetect.detect_spatial import calcDetectLoc, summarizeDetect, calcWpt, calcDetectIdx

# Set GHOSTVISION utils dir
USER_DIR = os.path.expanduser('~')
GV_UTILS_DIR = os.path.join(USER_DIR, '.ghostvision')

#===========================================
def crabpots_master_func(logfilename = '',
                        project_mode = 0,
                        script = '',
                        inFile = '',
                        sonFiles = '',
                        projDir = '',
                        coverage = False,
                        aoi = False,
                        max_heading_deviation = False,
                        max_heading_distance = False,
                        min_speed = False,
                        max_speed = False,
                        time_table = False,
                        tempC = 10,
                        nchunk = 500,
                        cropRange = 0,
                        exportUnknown = False,
                        fixNoDat = False,
                        threadCnt = 0,
                        pix_res_son = 0,
                        pix_res_map = 0,
                        x_offset = 0,
                        y_offset = 0,
                        tileFile = False,
                        egn = False,
                        egn_stretch = 0,
                        egn_stretch_factor = 1,
                        wcp = False,
                        wcm = False,
                        wcr = False,
                        wco = False,
                        sonogram_colorMap = 'Greys',
                        mask_shdw = False,
                        mask_wc = False,
                        spdCor = False,
                        maxCrop = False,
                        moving_window = False,
                        window_stride = 0.1,
                        USE_GPU = False,
                        remShadow = 0,
                        detectDep = 0,
                        smthDep = 0,
                        adjDep = 0,
                        pltBedPick = False,
                        rect_wcp = False,
                        rect_wcr = False,
                        rubberSheeting = True,
                        rectMethod = 'COG',
                        rectInterpDist = 50,
                        son_colorMap = 'Greys',
                        pred_sub = 0,
                        map_sub = 0,
                        export_poly = False,
                        map_predict = 0,
                        pltSubClass = False,
                        map_class_method = 'max',
                        mosaic_nchunk = 50,
                        mosaic = False,
                        map_mosaic = 0,
                        banklines = False,
                        rf_model = '',
                        gpxToHum = True,
                        sdDir = '',
                        confidence = 0.5,
                        iou_threshold = 0.5,
                        wptPrefix = '',
                        stride = 0,
                        export_image = False,
                        delete_image = False,
                        export_vid = False,
                        inference_track=False,
                        tracker_cnt = 1):
    

    ###############################################
    # Specify multithreaded processing thread count
    if threadCnt==0: # Use all threads
        threadCnt=cpu_count()
    elif threadCnt<0: # Use all threads except threadCnt; i.e., (cpu_count + (-threadCnt))
        threadCnt=cpu_count()+threadCnt
        if threadCnt<0: # Make sure not negative
            threadCnt=1
    elif threadCnt<1: # Use proportion of available threads
        threadCnt = int(cpu_count()*threadCnt)
        # Make even number
        if threadCnt % 2 == 1:
            threadCnt -= 1
    else: # Use specified threadCnt if positive
        pass

    if threadCnt>cpu_count(): # If more than total avail. threads, make cpu_count()
        threadCnt=cpu_count();
        print("\nWARNING: Specified more process threads then available, \nusing {} threads instead.".format(threadCnt))

    ####################################################
    # Check if sonObj pickle exists, append to metaFiles
    metaDir = os.path.join(projDir, "meta")
    print(metaDir)
    if os.path.exists(metaDir):
        metaFiles = sorted(glob(metaDir+os.sep+"*.meta"))
    else:
        sys.exit("No SON metadata files exist")
    del metaDir

    #############################################
    # Create a crabObj instance from pickle files
    crabObjs = []
    for meta in metaFiles:
        son = crabObj(meta) # Initialize mapObj()
        if son.beamName == 'ss_port' or son.beamName == 'ss_star':
            crabObjs.append(son) # Store mapObj() in mapObjs[]
    del meta, metaFiles

    ###
    # Copy model to /tmp/cache for roboflow
    if rf_model != '':
        import shutil
        tmp_model_dir = r'/tmp/cache'
        tmp_model_dir = os.path.join(tmp_model_dir, rf_model)

        rf_model_dir = os.path.join(GV_UTILS_DIR, 'models', rf_model)

        if os.path.exists(tmp_model_dir):
            shutil.rmtree(tmp_model_dir)
        
        shutil.copytree(rf_model_dir, tmp_model_dir)

    ##############
    # Do inference
    for son in crabObjs:

        # Get wcp folder
        wcp_dir_name = 'wcp_mw'
        wcp_dir = os.path.join(son.outDir, wcp_dir_name)

        out_dir_name = os.path.basename(wcp_dir)+'_results'
        outDir = os.path.join(os.path.dirname(wcp_dir), out_dir_name)

        channel = (son.beamName) #ss_port, ss_star, etc.
        projName = os.path.split(son.projDir)[-1]

        # Without tracking
        if not inference_track:
            print('\n\nNot Tracking Objects...\n')

            detect_csv = os.path.join(outDir, '{}_crabpot_detection_{}_ALL.csv'.format(projName, channel))

            # Inference
            son._detectCrabPots(rf_model=rf_model, in_dir=wcp_dir, out_dir=outDir, detect_csv=detect_csv, export_image=export_image, confidence=confidence, iou_threshold=iou_threshold)

            # Video
            if export_image and export_vid:

                # images = [img for img in os.listdir(image_folder) if img.endswith((".png", ".jpg", ".jpeg"))]
                images = [img for img in os.listdir(outDir) if img.endswith('.jpg') or img.endswith('.png') and channel in img]
                images.sort()

                vid_path = os.path.join(outDir, '{}_crabpot_detection_{}.mp4'.format(projName, channel))

                frame = cv2.imread(os.path.join(outDir, images[0]))
                height, width, layers = frame.shape

                video = cv2.VideoWriter(vid_path, cv2.VideoWriter_fourcc(*'mp4v'), 8, (width, height), )
                for image in images:
                    frame = cv2.imread(os.path.join(outDir, image))
                    video.write(frame)

                video.release()

                if delete_image:
                    for image in images:
                        # delet
                        os.remove(os.path.join(outDir, image))

        # With tracking
        if inference_track:
            print('\n\nTracking Objects...\n')

            detect_csv = os.path.join(outDir, '{}_crabpot_detection_{}_track_ALL.csv'.format(projName, channel))

            print(wcp_dir)
            print(os.path.exists(wcp_dir))
            print('confidence: {}\tiou: {}'.format(confidence, iou_threshold))

            if not os.path.exists(outDir):
                os.makedirs(outDir)
            # else:
            #     shutil.rmtree(outDir)
            #     os.makedirs(outDir)

            ##################
            # Create the video

            # images = [img for img in os.listdir(image_folder) if img.endswith((".png", ".jpg", ".jpeg"))]
            images = [img for img in os.listdir(wcp_dir) if img.endswith('.jpg') or img.endswith('.png') and channel in img]
            images.sort()

            vid_path = os.path.join(outDir, '{}_crabpot_detection_{}.mp4'.format(projName, channel))

            frame = cv2.imread(os.path.join(wcp_dir, images[0]))
            height, width, layers = frame.shape

            video = cv2.VideoWriter(vid_path, cv2.VideoWriter_fourcc(*'mp4v'), 10, (width, height), )
            for image in images:
                frame = cv2.imread(os.path.join(wcp_dir, image))
                video.write(frame)

            video.release()

            #######################
            # Do inference tracking
            son._detectTrackCrabPots(rf_model=rf_model, in_vid=vid_path, confidence=confidence, iou_threshold=iou_threshold, stride=window_stride, nchunk=nchunk)


        ###########################
        # Calculate mapped location

        if os.path.exists(detect_csv):
            # detect_csv = os.path.join(outDir, '{}_crabpot_detection_{}_track_ALL.csv'.format(projName, channel))
            detectDF = pd.read_csv(detect_csv)

            # Calculate ping index to get smoothed trackline data
            smthTrkFile = son.smthTrkFile
            detectDF = calcDetectIdx(smthTrkFile, detectDF, stride, son.nchunk)

            # Calculate target location
            beamName = son.beamName
            if rectMethod == 'Heading':
                cog = False
            else:
                cog = True
            detectDF = calcDetectLoc(beamName, detectDF, cog=cog)

            # Save all preds to csv
            detectDF.to_csv(detect_csv, index=False)

            if inference_track:
                # Summarize by target_id
                detectDF = summarizeDetect(detectDF)

                detectDF = detectDF.loc[detectDF['pred_cnt'] >= tracker_cnt]

            # Create wpt shapefile and GPX
            if len(detectDF)>0:
                projDir = son.projDir
                calcWpt(detectDF, outDir, projDir)
                
    # Delete model
    tmp_model_dir = r'/tmp/cache'
    tmp_model_dir = os.path.join(tmp_model_dir, rf_model)
    tmp_model_dir = os.path.dirname(tmp_model_dir)
    if os.path.exists(tmp_model_dir):
        import shutil
        shutil.rmtree(tmp_model_dir)


    return

#===========================================
def export_final_results(outDir: str,
                         projName: str):
    '''
    '''

    # Create Output Folder
    out = os.path.join(outDir, '0_GhostVision_FinalResults')
    if not os.path.exists(out):
        os.makedirs(out)

    #########################
    # Shapefile

    # Find all the shapefiles
    shps = glob(os.path.join(outDir, '**', '*.shp'), recursive=True)
    
    allShps = []
    for s in shps:
        s = gpd.read_file(s)
        allShps.append(s)

    gdf = gpd.GeoDataFrame(pd.concat(allShps, ignore_index=True))

    outShp = '{}_GhostVisionDetects.shp'.format(projName)
    outShp = os.path.join(out, outShp)

    gdf.to_file(outShp)

    #################
    # Raw CSV Results
    gdf.to_csv(outShp.replace('.shp', '.csv'), index=False)

    #####
    # GPX
    outGpx = outShp.replace('.shp', '.gpx')

    gdf = gdf.rename(columns={'tracker_id': 'name'})
    gdf = gdf[['name', 'geometry']]
    gdf.to_file(outGpx, 'GPX')

    return