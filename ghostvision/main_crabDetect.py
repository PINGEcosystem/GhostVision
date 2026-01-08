
'''

'''

import os, sys
import cv2
from ghostvision.class_crabObj_rf import crabObj

from joblib import Parallel, delayed, cpu_count
import numpy as np
import pandas as pd
import geopandas as gpd
from glob import glob

# # Debug
# detectPath = os.path.normpath('../PINGDetect')
# detectPath = os.path.abspath(detectPath)
# sys.path.insert(0, detectPath)

# tilePath = os.path.normpath('../PINGTile')
# tilePath = os.path.abspath(tilePath)
# sys.path.insert(0, tilePath)    

from pingdetect.detect_spatial import calcDetectLoc, summarizeDetect, calcWpt, calcDetectIdx
from pingtile.sonogramMovWin import doSonogramMovWin

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

        ##############################
        # Generate moving window tiles and videos
        print('\n\nGenerating Moving Window Tiles and Videos...\n')
        doSonogramMovWin(inDir=son.outDir,
                        projName=projName,
                        channel=channel,
                        sonMetaFile=son.sonMetaFile,
                        nchunk=nchunk,
                        stride=int(window_stride*nchunk),
                        tileType=['wcp'],
                        exportVid=True,
                        threadCnt=threadCnt)

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

            print(wcp_dir)
            print(os.path.exists(wcp_dir))
            print('confidence: {}\tiou: {}'.format(confidence, iou_threshold))

            if not os.path.exists(outDir):
                os.makedirs(outDir)
            # else:
            #     shutil.rmtree(outDir)
            #     os.makedirs(outDir)

            ##################
            # Get generated videos for tracking
            vid_dir = os.path.join(son.outDir, 'wcp_mw_results')
            vids = [os.path.join(vid_dir, f) for f in os.listdir(vid_dir) if f.endswith('.mp4') and channel in f]
            vids.sort()

            vids.sort()

            #######################
            # Do inference tracking on all videos
            son._detectTrackCrabPots(rf_model=rf_model, in_vids=vids, confidence=confidence, iou_threshold=iou_threshold, stride=window_stride, nchunk=nchunk)

            # Update detect_csv to the actual output file from tracker
            # The tracker saves with the name based on the first video
            detect_csv = os.path.join(vid_dir, os.path.basename(vids[0]).replace('.mp4', '_track_ALL.csv'))


        ###########################
        # Calculate mapped location

        # print(f"\n[DEBUG] Looking for detect_csv at: {detect_csv}")
        # print(f"[DEBUG] CSV exists: {os.path.exists(detect_csv)}")
        
        if os.path.exists(detect_csv):
            # detect_csv = os.path.join(outDir, '{}_crabpot_detection_{}_track_ALL.csv'.format(projName, channel))
            detectDF = pd.read_csv(detect_csv)
            # print(f"\n[DEBUG] Total detections loaded from {os.path.basename(detect_csv)}: {len(detectDF)}")
            # print(f"[DEBUG] Transect column exists: {'transect' in detectDF.columns}")
            # if 'transect' in detectDF.columns:
            #     print(f"[DEBUG] Unique transects: {sorted(detectDF['transect'].unique())}")
            #     print(f"[DEBUG] Transect value counts:\n{detectDF['transect'].value_counts().sort_index()}")

            # Calculate ping index to get smoothed trackline data
            smthTrkFile = son.smthTrkFile
            
            # Spatial indexing using stride-only mapping
            # Iterate over each transect
            finalDFS = []
            for transect_id in detectDF['transect'].unique():
                transectDF = detectDF[detectDF['transect'] == transect_id].copy()
                # print(f"[DEBUG] Processing transect {transect_id}: {len(transectDF)} detections")
                transectDF = calcDetectIdx(smthTrkFile, transectDF, stride, son.nchunk, transect_id=transect_id)
                # print(f"[DEBUG] After calcDetectIdx for transect {transect_id}: {len(transectDF)} detections")
                finalDFS.append(transectDF)
            detectDF = pd.concat(finalDFS, ignore_index=True)
            # print(f"[DEBUG] After concatenating all transects: {len(detectDF)} detections")

            # Calculate target location
            beamName = son.beamName
            if rectMethod == 'Heading':
                cog = False
            else:
                cog = True
            detectDF = calcDetectLoc(beamName, detectDF, cog=cog)
            # print(f"[DEBUG] After calcDetectLoc: {len(detectDF)} detections")

            # Add projName column
            detectDF['projName'] = projName

            # Save all preds to csv
            detectDF.to_csv(detect_csv, index=False)
            # print(f"[DEBUG] Saved {len(detectDF)} detections to CSV")

            if inference_track:
                # Summarize by target_id
                detectDF_before_summary = detectDF.copy()
                detectDF = summarizeDetect(detectDF)
                # print(f"[DEBUG] After summarizeDetect: {len(detectDF)} detections (was {len(detectDF_before_summary)})")

                detectDF_before_threshold = detectDF.copy()
                detectDF = detectDF.loc[detectDF['pred_cnt'] >= tracker_cnt]
                # print(f"[DEBUG] After pred_cnt threshold (>= {tracker_cnt}): {len(detectDF)} detections (was {len(detectDF_before_threshold)})")
            # Create wpt shapefile and GPX
            if len(detectDF)>0:
                projDir = son.projDir
                # print(f"\n[DEBUG] Creating shapefile/GPX with {len(detectDF)} detections")
                # print(f"[DEBUG] Output directory: {outDir}")
                # print(f"[DEBUG] Project directory: {projDir}")
                # Use the same confidence threshold for waypoint export to keep behavior consistent
                calcWpt(detectDF, outDir, projDir, threshold=confidence)
            else:
                # print(f"\n[DEBUG] No detections to export after filtering (DataFrame is empty)")
                pass
                
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
    # shps = glob(os.path.join(outDir, '**', '*.shp'), recursive=True)
    shps = glob(os.path.join(outDir, '**', '*_results', '*.shp'), recursive=True)

    if len(shps) == 0:
        # Nothing to export
        return

    # Harmonize CRS to WGS84 (EPSG:4326) for consistent merge and GPX output
    common_crs = 'EPSG:4326'
    allShps = []
    for shp_path in shps:
        df = gpd.read_file(shp_path)
        # Reproject if CRS is defined and different
        if df.crs is not None:
            try:
                df = df.to_crs(common_crs)
            except Exception:
                # If reprojection fails, keep original to avoid hard crash
                pass
        allShps.append(df)

    # Concatenate with a defined CRS
    gdf = gpd.GeoDataFrame(pd.concat(allShps, ignore_index=True), crs=common_crs)

    outShp = '{}_GhostVisionDetects.shp'.format(projName)
    outShp = os.path.join(out, outShp)

    gdf.to_file(outShp)

    #################
    # Raw CSV Results
    gdf.to_csv(outShp.replace('.shp', '.csv'), index=False)

    #####
    # GPX
    outGpx = outShp.replace('.shp', '.gpx')

    # Ensure GPX has a 'name' column
    if 'tracker_id' in gdf.columns:
        gdf = gdf.rename(columns={'tracker_id': 'name'})
    if 'name' not in gdf.columns:
        gdf['name'] = ''
    gdf = gdf[['name', 'geometry']]
    # GPX requires WGS84
    if gdf.crs is None or str(gdf.crs).upper() != 'EPSG:4326':
        try:
            gdf = gdf.to_crs('EPSG:4326')
        except Exception:
            pass
    gdf.to_file(outGpx, 'GPX')

    return