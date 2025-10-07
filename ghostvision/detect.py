

'''
Automated crab pot detection from Humminbird side imaging systems:

1) Extract sonar data from Humminbird .SON files (PING-Mapper)
2) Predict crab pots (Roboflow Inference Python API; model from Dr. T)
3) Calculate GPS coordinates (PING-Mapper)
'''

#============================================

# Imports
import sys, os
import shutil
import time, datetime
import json

from .version import __version__

# # Debug
# pingPath = os.path.normpath('../PINGMapper/pingmapper')
# pingPath = os.path.abspath(pingPath)
# sys.path.insert(0, pingPath)
# sys.path.insert(0, 'src')
# from funcs_common import *
# from main_readFiles import read_master_func
# from main_rectify import rectify_master_func

from pingmapper.funcs_common import *
from pingmapper.main_readFiles import read_master_func
from pingmapper.main_rectify import rectify_master_func

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PACKAGE_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.append(PACKAGE_DIR)

# from main_crabDetect import crabpots_master_func
from ghostvision.main_crabDetect import crabpots_master_func, export_final_results

from glob import glob

# Set GHOSTVISION utils dir
USER_DIR = os.path.expanduser('~')
GV_UTILS_DIR = os.path.join(USER_DIR, '.ghostvision')
rf_model_dir = os.path.join(GV_UTILS_DIR, 'models')

filter_time_csv = os.path.join(GV_UTILS_DIR, 'clip_table.csv')
filter_time_csv = os.path.normpath(filter_time_csv)

def detect_main(batch: bool=True):

    # For the logfile
    oldOutput = sys.stdout

    start_time = time.time()

    #============================================
    # Default Parameters - Not planned to change
    nchunk = 500
    rectMethod='COG'

    #============================================
    # Parameters

    # inDir = r'/mnt/c/Users/cbodine/Desktop/Crabpot_dev/recordings'
    # outDir = r'/mnt/c/Users/cbodine/Desktop/Crabpot_dev/'
    # projName = r'202509_test_newpackage'

    # rf_model = 'allcrabpotsources/11'  # Roboflow model name or path to local model

    # project_mode = 2
    # prefix = 'BEP2024_'
    # suffix = ''
    # gpxToHum = True
    # confidence = 0.5
    # iou_threshold = 0.1

    # cropRange = 25

    # moving_window = True
    # window_stride = 0.05 #0.05

    # export_image = True
    # export_vid = True

    # inference_track = True
    # track_cnt_thresh = 0.25 # Percent of time the target is seen in window to keep
    # tracker_cnt = int((nchunk / (window_stride*nchunk)) * track_cnt_thresh)

    # time_table = False

    #============================================

    # GUI
    # PySimpleGUI_License = 'e3yAJ9MVaOWANplCbmndNNl2VwHvlCwpZTSjIl6DIjkiRGpYc53aRty8aBWpJF1qdwGLlzv9bUiHILs3Inkyxpp5Yq2OVku8cg2ZVrJ7RNCQI66bMcTLcnyKMbTRMK57OCTPMGxGNtS8whirTBGTlLjxZEWg5DzWZdUXRUlLcDGfxnv7eiWB1jlOb6nqR8WTZ2XsJVzbabW19ouWI6j0oXiKN0Si4AwtI7iFw8iGTBmtFftjZEUxZMpYcLncNk0rIJj4oyisQq2uFCtqZnXWJvvqbEiCICsSIbkC5jhKbvWTVqM2YtX6Ni0XIJjloji1QEmU9Ak5ayWp5nlnIwi3wiiOQK279ytqcKGwFGuvepS6IH6iIOiYIGs7I4kYNQ13cY33RkvIbqWkVyypSKUOQoiZO2ijIFzaMNTEAp0bNxyWI1sLIwkRRjhZdNGBVoJkcZ3MNN1yZMWTQtihOiieIYyXMDDIIF0ILaTyAt36LKTREj5JI1iYwcixRgGuFk0BZGU5VZ4dciGUl3ykZ3XtMbilOMiBIhy1M5DtId1mL6T1A935LYTLEN5iI3iJwoirR8Wa12h5a0WtxkBNZdGiRJyYZXX9N5zZI2jSoZizYpmp9YkHaIWz5YluLTmcNXzNQmGZd0twYGW6l3sALZmTNWvubcSEItsPITk6lFQgQUWZRrkfcEmAVxz0c9y7IG6sILjZEYyzO8Cf4c0WLDj3QCwSLwjPEt2BMMi0J69p854e39898f71ea82d3a530f7a6ed8a02a4eea9ffd2c7b1279074b491c71b411f392e6d726a2d2f9dbf63388356cf4e083e358fe428852d676073e128607b9ad194c15e34a4feb463a749fd3295606caa293b823d102e854cd845b79b5ec5eaec0b2ef7f9cf0c87b2dfcad3f14cd0d66a2da97e6b38a535eb8707b4486c9802a4bfeb09703382e157449096f0e3551af9f444197cacb3f3d42187cea97ab61978985ddeecd086b9cb86c4ec1c08082d47b3ed0ae9c044d9aa65e5c9bf6e00238f78ed858cfdaf0021fb95d636e0cce84d84d2c2da7ac57f2e54fe793fce44a8b8abf96ce7c381f4b7eeb55dc4b68768e8172a4dffc1b683e62a108b2dfc2ef340dab058e6ee5c1f525f93e89d39258862f099987a8ec7022db5aecb5a58e81d02370d5717d18498ae58749aa5e463cf757ab7fa84efe49c1b770da397eef22423696ad433e7232646e279906bef084b21714ac5fc2af564a03ebc789123aed44531765b3e72c6165131feab68e35e0276a64760ee9abf043bece1e3cd148bcec97ab835395391387ff9d2b74a835a15ea5bac9c7e1218c217481a3999a91e037a138aaf5dddadb2247141242140b130e273aab5e1e6855fae8b7ee80d64be2d09a46f3d49555f53a7a849138fc3b9d2323658ea7e86a0039c40f3c15fd3647f99ec98232d9734a5933177c48c6575a1415e2808640cfb27773e728fe128b99757'
    # import PySimpleGUI as sg

    # inDirInit = r'/media/cbodine'
    # outDirInit = r'/media/cbodine/UDEL_Ubuntu/SynologyDrive/UDEL/Projects/CrabPots/2024_CrabPotRoundup/ModelPredictions/'

    # # inDirInit = r'/home/cbodine/Desktop/CrabPotDemo/20241011_Test/'
    # # outDirInit = r'/home/cbodine/Desktop/CrabPotDemo/20241011_Test/'

    # prefixDefault = ''
    # cropRangeDefault = 0
    # nchunk = 500
    # rect_wcp = False
    # egn=False
    # egn_stretch=1

    # # aoi = '/home/cbodine/Desktop/CrabPot_Workshop_LiveDemo/Rec00002/PINGMapper_aoi/PINGMapper_aoi_coverage.shp'


    #============================================

    # Default Values
    # Edit values below to change default values in gui
    primary_default_params = os.path.join(SCRIPT_DIR, "default_params.json")

    if not os.path.exists(primary_default_params):
        d = os.environ['CONDA_PREFIX']
        primary_default_params = os.path.join(d, 'ghostvision_config', 'default_params.json')
    
    default_params_file = os.path.join(GV_UTILS_DIR, "user_params.json")

    if not os.path.exists(default_params_file):
        default_params_file = primary_default_params
    with open(default_params_file) as f:
        default_params = json.load(f)

    # Make sure all params in user params
    with open(primary_default_params) as f:
        primary_defaults = json.load(f)

    for k, v in primary_defaults.items():
        if k not in default_params:
            default_params[k] = v
    

    ############
    # Set Up GUI

    import PySimpleGUI as sg

    layout = []

    # Title #
    title = sg.Text("GhostVision", font=("Helvetica", 24), justification="center", size=(100,1))
    version = sg.Text("ver. {}".format(__version__), font=("Helvetica", 8), justification="center", size=(100,1))

    layout.append([title])
    layout.append([version])

    ####################
    # General Parameters
    text_io = sg.Text('I/O\n', font=("Helvetica", 14, "underline"))


    if batch:
        text_input = sg.Text('Parent Folder of Recordings to Process')
        # in_input = sg.In(key='inDir', size=(80,1))
        in_input = sg.In(key='inDir', size=(80,1), default_text=default_params['inDir'])
        browse_input = sg.FolderBrowse(initial_folder=(default_params['inDir']))

    else:
        text_input = sg.Text('Recording to Process')
        # in_input = sg.In(key='inFile', size=(80,1))
        in_input = sg.In(key='inFile', size=(80,1), default_text=default_params['inFile'])
        browse_input = sg.FileBrowse(file_types=(("Sonar File", "*.DAT *.sl2 *.sl3 *.RSD *.svlog") ), initial_folder=os.path.dirname(default_params['inFile']))
        # browse_input = sg.FileBrowse(file_types=(("Sonar File", "*.DAT *.sl2 *.sl3 *.svlog") ), initial_folder=os.path.dirname(default_params['inFile']))

    # Add to layout
    layout.append([text_io])
    layout.append([text_input])
    layout.append([in_input, browse_input])

    ###################
    # Output parameters
    text_output = sg.Text('Output Folder')
    # in_output = sg.In(key='proj', size=(80,1))
    in_output = sg.In(key='proj', size=(80,1), default_text=default_params['proj'])
    browse_output = sg.FolderBrowse(initial_folder=os.path.dirname(default_params['proj']))

    # Overwrite
    check_overwrite = sg.Checkbox('Overwrite Existing Project', key='project_mode', default=default_params['project_mode'])


    # Add to layout
    layout.append([text_output])
    layout.append([in_output, browse_output])
    layout.append([check_overwrite])

    ##############
    # Project Name
    if batch:
        text_prefix = sg.Text('Project Name Prefix:', size=(20,1))
        in_prefix = sg.Input(key='prefix', size=(10,1))

        text_suffix = sg.Text('Project Name Suffix:', size=(20,1))
        in_suffix = sg.Input(key='suffix', size=(10,1))

        # Add to layout
        layout.append([text_prefix, in_prefix, sg.VerticalSeparator(), text_suffix, in_suffix])

    else:
        text_project = sg.Text('Project Name', size=(15,1))
        in_project = sg.InputText(key='projName', size=(50,1), default_text=os.path.basename(default_params['projDir']))

        # Add to layout
        layout.append([text_project, in_project])


    # Waypoint prefix #
    wpt_label = sg.Text('Waypoint Prefix:', size=(20,1))
    wpt_input = sg.Input(key='wptPrefix', size=(10,1), default_text=default_params['wptPrefix'])
    wpt_check = sg.Checkbox('Export Detections to Humminbird SD Card', key='gpxToHum', default=default_params['gpxToHum'])

    # Chunk
    text_chunk = sg.Text('Chunk Size', size=(20,1))
    in_chunk = sg.Input(key='nchunk', default_text=default_params['nchunk'], size=(10,1))

    
    # layout.append([text_prefix, in_prefix, sg.VerticalSeparator(), text_suffix, in_suffix])
    layout.append([wpt_label, wpt_input, sg.VerticalSeparator(), wpt_check])
    layout.append([text_chunk, in_chunk])


    ###########
    # Filtering
    text_filtering = sg.Text('Filter Sonar Log\n', font=("Helvetica", 14, "underline"))

    # Cropping
    text_crop = sg.Text('Crop Range [m]', size=(22,1))
    in_crop = sg.Input(key='cropRange', default_text=default_params['cropRange'], size=(10,1))

    # Heading
    text_heading = sg.Text('Max. Heading Deviation [deg]:', size=(22,1))
    in_heading = sg.Input(key='max_heading_deviation', default_text=default_params['max_heading_deviation'], size=(10,1))
    text_distance = sg.Text('Distance [m]:', size=(15,1))
    in_distance = sg.Input(key='max_heading_distance', default_text=default_params['max_heading_distance'], size=(10,1))

    # Speed
    text_speed_min = sg.Text('Min. Speed [m/s]:', size=(22,1))
    in_speed_min = sg.Input(key='min_speed', default_text=default_params['min_speed'], size=(10,1))
    text_speed_max = sg.Text('Max. Speed [m/s]:', size=(15,1))
    in_speed_max = sg.Input(key='max_speed', default_text=default_params['max_speed'], size=(10,1))

    # AOI
    text_aoi = sg.Text('AOI')
    in_aoi = sg.In(size=(80,1))
    browse_aoi = sg.FileBrowse(key='aoi', file_types=(("Shapefile", "*.shp"), (".plan File", "*.plan")), initial_folder=os.path.dirname(default_params['aoi']))

    # Time table
    button_time_table = sg.Button('Edit Table')
    check_time_load = sg.Checkbox('Filter by Time', key='filter_table', default=default_params['filter_table'])

    # Add to layout
    layout.append([sg.HorizontalSeparator()])
    layout.append([text_filtering])
    layout.append([text_crop, in_crop])
    layout.append([text_heading, in_heading, sg.VerticalSeparator(), text_distance, in_distance])
    layout.append([text_speed_min, in_speed_min, sg.VerticalSeparator(), text_speed_max, in_speed_max])
    layout.append([text_aoi])
    layout.append([in_aoi, browse_aoi])
    layout.append([check_time_load, button_time_table])


    ######################
    # Position Corrections

    # Position text
    text_position = sg.Text('Position Corrections\n', font=("Helvetica", 14, "underline"))

    # X offset
    text_x_offset = sg.Text('Transducer Offset [X]:', size=(22,1))
    in_x_offset = sg.Input(key='x_offset', default_text=default_params['x_offset'], size=(10,1))
    
    # Y offset
    text_y_offset = sg.Text('Transducer Offset [Y]:', size=(22,1))
    in_y_offset = sg.Input(key='y_offset', default_text=default_params['y_offset'], size=(10,1))

    # Add to layout
    layout.append([sg.HorizontalSeparator()])
    layout.append([text_position])
    layout.append([text_x_offset, in_x_offset, sg.VerticalSeparator(), text_y_offset, in_y_offset])


    ##################
    # Detection Params

    text_detect = sg.Text('Detection Parameters\n', font=("Helvetica", 14, "underline"))


    # Model Selection #
    avail_models = get_avail_models()
    model_label = sg.Text("Model Selection:", size=(20, 1), font=("Helvetica", 12), justification="left")
    model_list = sg.Combo(avail_models, key='rf_model', default_value=default_params['rf_model'])

    
    # Confidence & IoU #
    conf_label = sg.Text('Confidence Threshold', size=(20,1))
    conf_slider = sg.Slider((0,1), key='confidence', default_value=default_params['confidence'], resolution=0.05, tick_interval=0.25, orientation='horizontal')
    iou_label = sg.Text('IoU Threshold', size=(20,1))
    iou_slider = sg.Slider((0,1), key='iou_threshold', default_value=default_params['iou_threshold'], resolution=0.05, tick_interval=0.25, orientation='horizontal')

    # Moving Window #
    check_mov_win = sg.Checkbox('Moving Window', key='moving_window', default=default_params['moving_window'], enable_events=True)
    if default_params['moving_window'] == True:
        mov_win_status = False
    else:
        mov_win_status = True
    text_mov_win = sg.Text('Window Stride', size=(20,1))
    slide_mov_win = sg.Slider((0,1), key='window_stride', default_value=default_params['window_stride'], resolution=0.025, tick_interval=0.25, orientation='horizontal', disabled=mov_win_status)

    col_detect_1 = sg.Column([[model_label, model_list], 
                              [check_mov_win],
                              [text_mov_win, slide_mov_win]], 
                              vertical_alignment='top')
    
    col_detect_2 = sg.Column([[conf_label, conf_slider],
                              [iou_label, iou_slider]],
                              vertical_alignment='top')

    # Add to layout
    layout.append([sg.HorizontalSeparator()])
    layout.append([text_detect])
    layout.append([col_detect_1, sg.VerticalSeparator(), col_detect_2])


    ########################
    # Object Tracking Params

    text_track = sg.Text('Object Tracking Parameters\n', font=("Helvetica", 14, "underline"))

    # Inference Tracking #
    check_track = sg.Checkbox('Track Objects', key='inference_track', default=default_params['inference_track'], enable_events=True)
    if default_params['inference_track'] == True:
        track_status = False
    else:
        track_status = True
    text_track_thresh = sg.Text('Tracking Threshold', size=(20,1))
    slide_track = sg.Slider((0,1), key='track_cnt_thresh', default_value=default_params['track_cnt_thresh'], resolution=0.05, tick_interval=0.25, orientation='horizontal', disabled=track_status)

    # Add to layout
    layout.append([sg.HorizontalSeparator()])
    layout.append([text_track])
    layout.append([check_track, sg.VerticalSeparator(), text_track_thresh, slide_track])

    #########
    # Exports

    text_exports = sg.Text('Export Options\n', font=("Helvetica", 14, "underline"))

    # Export Image
    check_image = sg.Checkbox('Export Detection Images', key='export_image', default=default_params['export_image'])
    # Export Video
    check_video = sg.Checkbox('Export Detection Videos', key='export_vid', default=default_params['export_vid'])

    # Add to layout
    layout.append([sg.HorizontalSeparator()])
    layout.append([text_exports])
    layout.append([check_image, sg.VerticalSeparator(), check_video])

    #####################
    # Submit/quit buttons
    layout.append([sg.HorizontalSeparator()])
    layout.append([sg.Push(), sg.Submit(), sg.Quit(), sg.Button('Save Defaults'), sg.Push()])

    layout2 =[[sg.Column(layout, scrollable=True,  vertical_scroll_only=True, size_subsample_height=2)]]
    window = sg.Window('GhostVision', layout2, resizable=True)

    while True:
        event, values = window.read()
        if event == "Quit" or event == 'Submit':
            break

        if event == "Save Defaults":
            from pingmapper.funcs_common import saveDefaultParams
            user_params = os.path.join(GV_UTILS_DIR, "user_params.json")
            saveDefaultParams(values, user_params)

        if event == 'Edit Table':
            from pingmapper.funcs_common import clip_table
            clip_table(filter_time_csv)

        if event == 'moving_window':
            if values['moving_window'] == True:
                window['window_stride'].update(disabled=False)
            else:
                window['window_stride'].update(disabled=True)

        if event == 'inference_track':
            if values['inference_track'] == True:
                window['track_cnt_thresh'].update(disabled=False)
            else:
                window['track_cnt_thresh'].update(disabled=True)

    window.close()
    #########
    # End GUI

    if event == "Quit":
        sys.exit()

    outDir = os.path.normpath(values['proj'])

    if batch:
        inDir = os.path.normpath(values['inDir'])

    #################################
    # Convert parameters if necessary

    if values['filter_table']:
        time_table = filter_time_csv
    else:
        time_table = False

    # AOI
    aoi = values['aoi']
    if aoi == '':
        aoi = False

    #============================================

    # params = {
    #     'project_mode':int(values['project_mode']),
    #     'nchunk': int(values['nchunk']),
    #     'aoi': values['aoi'],
    #     'cropRange': float(values['cropRange']),
    #     'threadCnt':0.5,
    #     'aoi':aoi,
    #     'max_heading_deviation':float(values['max_heading_deviation']),
    #     'max_heading_distance':float(values['max_heading_distance']),
    #     'min_speed':float(values['min_speed']),
    #     'max_speed':float(values['max_speed']),
    #     'time_table':time_table,
    #     # 'rect_wcp': values['rect_wcp'],
    #     'x_offset':float(values['x_offset']),
    #     'y_offset':float(values['y_offset']),
    #     'moving_window': bool(values['moving_window']),
    #     'window_stride': float(values['window_stride']),
    #     'wcp':True,
    #     'rectMethod':rectMethod
    # }

    # globals().update(params)    

    #============================================

    # Find all DAT and SON files in all subdirectories of inDir
    if batch:
        # Find all DAT and SON files in all subdirectories of inDir
        inFiles=[]
        for root, dirs, files in os.walk(inDir):
            if '__MACOSX' not in root:
                for file in files:
                    if file.endswith('.DAT') or file.endswith('.sl2') or file.endswith('.sl3') or file.endswith('.RSD') or file.endswith('.svlog'):
                        inFiles.append(os.path.join(root, file))

        inFiles = sorted(inFiles)

    else:
        inFiles = [values['inFile']]
    
    
    
    
    inFiles = inFiles[:2] # for testing



    #============================================
    # # Copy inFiles to outDir
    # if not os.path.exists(outDir):
    #     os.mkdir(outDir)

    # recordings = os.path.join(outDir, 'recordings')
    # if not os.path.exists(recordings):
    #     os.mkdir(recordings)

    #     for f in inFiles:
    #         shutil.copy(f, recordings)

    #         if f.endswith('.DAT'):
    #             sonPath = f.split('.DAT')[0]
    #             shutil.copytree(sonPath, os.path.join(recordings, os.path.basename(sonPath)))

    # # Update inFiles to new location
    # inFiles = [os.path.join(recordings, os.path.basename(f)) for f in inFiles]

    for i, f in enumerate(inFiles):
        print(i, ":", f)

    #============================================
    for datFile in inFiles:
        logfilename = 'log_'+time.strftime("%Y-%m-%d_%H%M")+'.txt'

        try:
            copied_script_name = os.path.basename(__file__).split('.')[0]+'_'+time.strftime("%Y-%m-%d_%H%M")+'.py'
            script = os.path.abspath(__file__)

            start_time_iter = time.time()  

            inPath = os.path.dirname(datFile)
            inFile = datFile
            recName = '.'.join(os.path.basename(inFile).split('.')[:-1])

            try:
                sonPath = inFile.split('.DAT')[0]
                sonFiles = sorted(glob(sonPath+os.sep+'*.SON'))
                # Remove sonfiles that are not port or starboard
                sonFiles = [s for s in sonFiles if os.path.basename(s) in ['B002.SON', 'B003.SON']]
            except:
                sonFiles = ''

            if batch:
                recName = values['prefix'] + recName + values['suffix']

                projDir = os.path.join(outDir, recName)

            else:
                projDir = os.path.join(os.path.normpath(values['proj']), values['projName'])

            #============================================

            # =========================================================
            # Determine project_mode
            project_mode = int(values['project_mode'])

            print(project_mode)
            if project_mode == 0:
                # Create new project
                if not os.path.exists(projDir):
                    os.mkdir(projDir)
                else:
                    projectMode_1_inval()

            elif project_mode == 1:
                # Overwrite existing project
                if os.path.exists(projDir):
                    shutil.rmtree(projDir)

                os.mkdir(projDir)        

            elif project_mode == 2:
                # Update project
                # Make sure project exists, exit if not.
                
                if not os.path.exists(projDir):
                    projectMode_2_inval()

            # =========================================================
            # For logging the console output

            logdir = os.path.join(projDir, 'logs')
            if not os.path.exists(logdir):
                os.makedirs(logdir)

            logfilename = os.path.join(logdir, logfilename)

            sys.stdout = Logger(logfilename)

            # =========================================================
            # Copy datFile to project directory
            outDatDir = os.path.join(projDir, 'recordings')
            if not os.path.exists(outDatDir):
                os.mkdir(outDatDir)

            outDatFile = os.path.join(outDatDir, os.path.basename(datFile))

            if not os.path.exists(outDatFile):
                shutil.copy(datFile, outDatFile)

                if datFile.endswith('.DAT'):
                    sonPath = datFile.split('.DAT')[0]
                    outSonPath = os.path.join(outDatDir, os.path.basename(sonPath))
                    shutil.copytree(sonPath, outSonPath)

            inFile = outDatFile

            #============================================
            # Parameters

            params = {
                'sonFiles': sonFiles,
                'logfilename': logfilename,
                'script': [script, copied_script_name],
                'projDir': projDir,
                'inFile': inFile,
                'project_mode':int(values['project_mode']),
                'nchunk': int(values['nchunk']),
                'aoi': values['aoi'],
                'cropRange': float(values['cropRange']),
                'threadCnt':0.5,
                'aoi':aoi,
                'max_heading_deviation':float(values['max_heading_deviation']),
                'max_heading_distance':float(values['max_heading_distance']),
                'min_speed':float(values['min_speed']),
                'max_speed':float(values['max_speed']),
                'time_table':time_table,
                # 'rect_wcp': values['rect_wcp'],
                'x_offset':float(values['x_offset']),
                'y_offset':float(values['y_offset']),
                'moving_window': bool(values['moving_window']),
                'window_stride': float(values['window_stride']),
                'wcp':True,
                'rectMethod':rectMethod
            }

            #============================================

            print('\n\n', '***User Parameters***')
            for k,v in params.items():
                print("| {:<20s} : {:<10s} |".format(k, str(v)))


            # try:

            print('sonPath',sonPath)
            print('\n\n\n+++++++++++++++++++++++++++++++++++++++++++')
            print('+++++++++++++++++++++++++++++++++++++++++++')
            print('***** Working On *****')
            print(inFile)
            print('Start Time: ', datetime.datetime.now().strftime('%Y-%m-%d %H:%M'))

            print('\n===========================================')
            print('===========================================')
            print('***** READING *****')
            read_master_func(**params)

            print('\n===========================================')
            print('===========================================')
            print('***** RECTIFYING *****')
            rectify_master_func(**params)

            params['rf_model'] = values['rf_model']
            params['gpxToHum'] = values['gpxToHum']
            params['sdDir'] = inDir
            params['confidence'] = values['confidence']
            params['iou_threshold'] = values['iou_threshold']

            # Unique Waypoint name
            recording = os.path.basename(inFile)
            recording = recording.split('.')[0]
            recording = int(recording[3:])
            # wptPrefix = values['wptPrefix']
            wptPrefix = 'wpt'
            wptPrefix = '{}_{}'.format(wptPrefix, recording)
            params['wptPrefix'] = wptPrefix
            window_stride = float(values['window_stride'])
            params['stride'] = int(window_stride*nchunk)
                

            params['export_vid'] = values['export_vid']
            params['export_image'] = values['export_image']
            params['inference_track'] = values['inference_track']
            params['tracker_cnt'] = ((nchunk / (window_stride*nchunk)) * values['track_cnt_thresh'])

            if params['export_vid'] and not params['export_image']:
                params['export_image'] = True
                params['delete_image'] = True

            print('\n\n', '***User Detection Parameters***')
            for k,v in params.items():
                print("| {:<20s} : {:<10s} |".format(k, str(v)))
            

            print('\n===========================================')
            print('===========================================')
            print('***** DETECTING CRAB POTS *****')
            crabpots_master_func(**params)

            print("\n\nTotal Processing Time: ",datetime.timedelta(seconds = round(time.time() - start_time_iter, ndigits=0)))

            sys.stdout.log.close()
                
        except Exception as Argument:
            unableToProcessError(logfilename)
            print('\n\nCould not process:', datFile)

        sys.stdout = oldOutput

        # sys.exit()

    print('\n===========================================')
    print('===========================================')
    print('***** EXPORTING FINAL RESULTS *****')
    export_final_results(outDir, os.path.basename(outDir))

    print("\n\nGrand Total Processing Time: ",datetime.timedelta(seconds = round(time.time() - start_time, ndigits=0)))

def get_avail_models():

    # Get projects in directory
    projects = os.listdir(rf_model_dir)

    
    # Find all folders and subfolders in rf_model_dir
    avail_models = []
    for proj in projects:
        versions = os.listdir(os.path.join(rf_model_dir, proj))

        for v in versions:
            avail_models.append('{}/{}'.format(proj, v))

    return avail_models

if __name__ == "__main__":
    detect_main()