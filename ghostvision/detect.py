

'''
Automated crab pot detection from Humminbird side imaging systems:

1) Extract sonar data from Humminbird .SON files (PING-Mapper)
2) Predict crab pots (Roboflow Inference Python API; model from Dr. T)
3) Calculate GPS coordinates (PING-Mapper)
'''

#============================================

# Imports
import sys
import os
import shutil
import time
import datetime
import json
import importlib
import csv
import numpy as np
import re
import textwrap
from PIL import Image
import requests
import zipfile
import FreeSimpleGUI as sg
import glob

from .version import __version__

# Debug
pingPath = os.path.normpath('../PINGMapper')
pingPath = os.path.abspath(pingPath)
sys.path.insert(0, pingPath)

pingverterPath = os.path.normpath('../PINGVerter')
pingverterPath = os.path.abspath(pingverterPath)
if os.path.exists(pingverterPath) and pingverterPath not in sys.path:
    sys.path.insert(0, pingverterPath)

from pingmapper.funcs_common import *

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

SUPPORTED_EXTS = ('.DAT', '.sl2', '.sl3', '.RSD', '.svlog', '.jsf', '.xtf')

# Match PingMapper tooltip behavior so hover text is readable and less flickery.
sg.set_options(tooltip_time=500, tooltip_offset=(18, 18))


def ml_tip(text, width=62):
    return textwrap.fill(text, width=width, break_long_words=False)


DEPTH_MODE_OPTIONS = {
    'Instrument/Metadata': 0,
    'Auto (ML with threshold fallback)': 1,
    'Binary Threshold': 2,
}


def _normalize_depth_mode_value(raw_value):
    if raw_value in DEPTH_MODE_OPTIONS:
        return raw_value

    try:
        numeric_value = int(raw_value)
    except (TypeError, ValueError):
        numeric_value = None

    numeric_to_label = {
        0: 'Instrument/Metadata',
        1: 'Auto (ML with threshold fallback)',
        2: 'Binary Threshold',
    }
    if numeric_value is None:
        return 'Binary Threshold'
    return numeric_to_label.get(numeric_value, 'Binary Threshold')


def _fit_window_to_screen(window, width_ratio=0.72, height_ratio=0.85):
    root = getattr(window, 'TKroot', None)
    if root is None:
        return

    root.update_idletasks()

    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    requested_width = root.winfo_reqwidth()
    requested_height = root.winfo_reqheight()

    max_width = max(900, int(screen_width * width_ratio))
    max_height = max(700, int(screen_height * height_ratio))
    target_width = min(requested_width, max_width)
    target_height = min(requested_height, max_height)

    offset_x = max((screen_width - target_width) // 2, 0)
    offset_y = max((screen_height - target_height) // 6, 0)
    root.geometry(f'{target_width}x{target_height}+{offset_x}+{offset_y}')


def _get_pingmapper_dowork():
    try:
        return importlib.import_module('pingmapper.doWork').doWork
    except Exception:
        pingmapper_repo = os.path.normpath(os.path.join(SCRIPT_DIR, '..', '..', 'PINGMapper'))
        if pingmapper_repo not in sys.path and os.path.exists(pingmapper_repo):
            sys.path.insert(0, pingmapper_repo)
        try:
            return importlib.import_module('pingmapper.doWork').doWork
        except Exception as exc:
            raise ImportError(
                'Could not import pingmapper.doWork.doWork. GhostVision uses PingMapper for this step; '
                'see the chained exception for the exact PingMapper dependency/import failure.'
            ) from exc


def _normalize_wcp_tiles(tile_paths):
    beam_groups = {}
    for path in tile_paths:
        match = re.search(r'wcp_(port|star)', path)
        if match:
            beam = match.group(1)
            beam_groups.setdefault(beam, []).append(path)

    for beam, paths in beam_groups.items():
        minval, maxval = None, None
        for path in paths:
            image = Image.open(path)
            array = np.array(image)
            vmin, vmax = array.min(), array.max()
            minval = vmin if minval is None else min(minval, vmin)
            maxval = vmax if maxval is None else max(maxval, vmax)

        if minval == maxval:
            continue

        for path in paths:
            image = Image.open(path)
            array = np.array(image)
            normalized = ((array - minval) / (maxval - minval) * 65535).astype(np.uint16)
            Image.fromarray(normalized).save(path)

        print(f"Normalized {len(paths)} tiles for beam '{beam}' (min={minval}, max={maxval})")

def detect_main(batch: bool=True):

    # For the logfile
    oldOutput = sys.stdout

    start_time = time.time()

    #============================================
    # Default Parameters - Not planned to change
    nchunk = 500
    rectMethod='COG'

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

    import FreeSimpleGUI as sg

    layout = []

    # Title #
    title = sg.Text("GhostVision", font=("Helvetica", 24), justification="center")
    version = sg.Text("ver. {}".format(__version__), font=("Helvetica", 8), justification="center")

    layout.append([sg.Push(), title, sg.Push()])
    layout.append([sg.Push(), version, sg.Push()])

    ####################
    # General Parameters
    text_io = sg.Text('I/O\n', font=("Helvetica", 14, "underline"))

    tip_input = ml_tip('Sonar log file to process. Supports .DAT (Humminbird), .sl2/.sl3 (Lowrance), .RSD (Garmin), .svlog (Cerulean), .jsf, and .xtf.')
    tip_output = ml_tip('Destination folder for all processed outputs. Avoid cloud drives such as OneDrive or Google Drive.')
    tip_overwrite = ml_tip('If checked, overwrite an existing project folder with the same name. If unchecked, GhostVision creates a new folder.')
    tip_project_name = ml_tip('Unique name for this processing project. A folder with this name is created inside the output folder.')
    tip_prefix = ml_tip('Prefix for batch project names. Useful for organizing multiple batch runs.')
    tip_suffix = ml_tip('Suffix for batch project names. Useful for organizing multiple batch runs.')
    tip_preserve_subdirs = ml_tip('Keep the input subdirectory structure inside the output folder when processing a batch directory.')
    tip_egn = ml_tip('Enable empirical gain normalization before detection export to reduce range-dependent intensity falloff.')
    tip_wpt_prefix = ml_tip('Prefix used when naming exported detection waypoints.')
    tip_hum_export = ml_tip('Export detections in a Humminbird-compatible GPX layout for copying to an SD card.')
    tip_crop = ml_tip('Crop sonar returns beyond this range in meters. 0 disables cropping.')
    tip_heading = ml_tip('Filter records based on vessel heading deviation over distance. 0 disables heading filtering.')
    tip_distance = ml_tip('Distance in meters over which heading deviation is calculated.')
    tip_speed_min = ml_tip('Minimum vessel speed filter in m/s. 0 disables the minimum-speed filter.')
    tip_speed_max = ml_tip('Maximum vessel speed filter in m/s. 0 disables the maximum-speed filter.')
    tip_aoi = ml_tip('Optional polygon shapefile or .plan file to spatially filter sonar records to an area of interest.')
    tip_time_filter = ml_tip('Filter sonar records by time range using the editable clip table.')
    tip_time_table = ml_tip('Open the clip table editor used when Filter by Time is enabled.')
    tip_x_offset = ml_tip('X offset in meters. Positive is toward the bow; negative is toward the stern.')
    tip_y_offset = ml_tip('Y offset in meters. Positive is toward starboard; negative is toward port.')
    tip_depth_mode = ml_tip('Depth method used by PINGMapper before GhostVision inference. Instrument/Metadata uses logged depth values; Auto requests ML depth detection with fallback logic in PINGMapper; Binary Threshold uses the non-ML threshold picker directly.')
    tip_model = ml_tip('Select the GhostVision detection model to run against the exported sonar imagery.')
    tip_confidence = ml_tip('Minimum score required to keep a detection. Without tracking this is the model confidence threshold; with tracking enabled it is applied to the alpha-weighted combined score.')
    tip_alpha = ml_tip('Weight used when combining confidence and persistence for tracked detections. A value of 1.0 prioritizes confidence only; 0.0 prioritizes persistence only.')
    tip_iou = ml_tip('Intersection-over-union threshold used during non-maximum suppression to remove overlapping detections.')
    tip_moving_window = ml_tip('Run inference with overlapping windows instead of single-pass tiles. This can improve detections near tile edges.')
    tip_window_stride = ml_tip('Stride fraction for the moving window. Smaller values increase overlap and runtime; larger values reduce overlap.')
    tip_track = ml_tip('Track detections across adjacent frames/tiles to merge repeated observations of the same object.')
    tip_track_threshold = ml_tip('Minimum tracking consistency threshold used to keep tracked objects in the final results.')
    tip_export_image = ml_tip('Export still images showing detections and annotations.')
    tip_export_video = ml_tip('Export videos showing detections and annotations across the processed sonar imagery.')


    if batch:
        text_input = sg.Text('Parent Folder of Recordings to Process')
        # in_input = sg.In(key='inDir', size=(80,1))
        in_input = sg.In(key='inDir', size=(80,1), default_text=default_params['inDir'], tooltip=tip_input)
        browse_input = sg.FolderBrowse(initial_folder=(default_params['inDir']))

    else:
        text_input = sg.Text('Recording to Process')
        # in_input = sg.In(key='inFile', size=(80,1))
        in_input = sg.In(key='inFile', size=(80,1), default_text=default_params['inFile'], tooltip=tip_input)
        browse_input = sg.FileBrowse(file_types=(("Sonar File", "*.DAT *.sl2 *.sl3 *.RSD *.svlog *.jsf *.xtf") ), initial_folder=os.path.dirname(default_params['inFile']))
        # browse_input = sg.FileBrowse(file_types=(("Sonar File", "*.DAT *.sl2 *.sl3 *.svlog") ), initial_folder=os.path.dirname(default_params['inFile']))

    # Add to layout
    layout.append([text_io])
    layout.append([text_input])
    layout.append([in_input, browse_input])

    ###################
    # Output parameters
    text_output = sg.Text('Output Folder')
    # in_output = sg.In(key='proj', size=(80,1))
    in_output = sg.In(key='proj', size=(80,1), default_text=default_params['proj'], tooltip=tip_output)
    browse_output = sg.FolderBrowse(initial_folder=os.path.dirname(default_params['proj']))

    # Overwrite
    check_overwrite = sg.Checkbox('Overwrite Existing Project', key='project_mode', default=default_params['project_mode'], tooltip=tip_overwrite)


    # Add to layout
    layout.append([text_output])
    layout.append([in_output, browse_output])
    layout.append([check_overwrite])

    ##############
    # Project Name

    if batch:
        text_prefix = sg.Text('Project Name Prefix:', size=(20,1))
        in_prefix = sg.Input(key='prefix', size=(10,1), tooltip=tip_prefix)

        text_suffix = sg.Text('Project Name Suffix:', size=(20,1))
        in_suffix = sg.Input(key='suffix', size=(10,1), tooltip=tip_suffix)

        check_preserve_subdirs = sg.Checkbox(
            'Preserve Input Subdirectory Structure',
            key='preserve_subdirs',
            default=default_params.get('preserve_subdirs', False),
            tooltip=tip_preserve_subdirs,
        )

        # Add to layout
        layout.append([text_prefix, in_prefix, sg.VerticalSeparator(), text_suffix, in_suffix])
        layout.append([check_preserve_subdirs])

    else:
        text_project = sg.Text('Project Name', size=(15,1))
        in_project = sg.InputText(key='projName', size=(50,1), default_text=os.path.basename(default_params['projDir']), tooltip=tip_project_name)

        # Add to layout
        layout.append([text_project, in_project])

    # EGN Option
    text_egn = sg.Text('Apply EGN', size=(20,1))
    check_egn = sg.Checkbox('Enable EGN', key='egn', default=default_params.get('egn', True), tooltip=tip_egn)
    layout.append([text_egn, check_egn])


    # Waypoint prefix #
    wpt_label = sg.Text('Waypoint Prefix:', size=(20,1))
    wpt_input = sg.Input(key='wptPrefix', size=(10,1), default_text=default_params['wptPrefix'], tooltip=tip_wpt_prefix)
    wpt_check = sg.Checkbox('Export Detections to Humminbird SD Card', key='gpxToHum', default=default_params['gpxToHum'], tooltip=tip_hum_export)

    # # Chunk
    # text_chunk = sg.Text('Chunk Size', size=(20,1))
    # in_chunk = sg.Input(key='nchunk', default_text=default_params['nchunk'], size=(10,1))

    
    # layout.append([text_prefix, in_prefix, sg.VerticalSeparator(), text_suffix, in_suffix])
    layout.append([wpt_label, wpt_input, sg.VerticalSeparator(), wpt_check])
    # layout.append([text_chunk, in_chunk])


    ###########
    # Filtering
    text_filtering = sg.Text('Filter Sonar Log\n', font=("Helvetica", 14, "underline"))

    # Cropping
    text_crop = sg.Text('Crop Range [m]', size=(22,1))
    in_crop = sg.Input(key='cropRange', default_text=default_params['cropRange'], size=(10,1), tooltip=tip_crop)

    text_depth_mode = sg.Text('Depth Method', size=(22,1))
    depth_mode_value = _normalize_depth_mode_value(default_params.get('detectDep', 'Binary Threshold'))
    depth_mode_combo = sg.Combo(list(DEPTH_MODE_OPTIONS.keys()), key='detectDep', default_value=depth_mode_value, readonly=True, tooltip=tip_depth_mode)

    # Heading
    text_heading = sg.Text('Max. Heading Deviation [deg]:', size=(22,1))
    in_heading = sg.Input(key='max_heading_deviation', default_text=default_params['max_heading_deviation'], size=(10,1), tooltip=tip_heading)
    text_distance = sg.Text('Distance [m]:', size=(15,1))
    in_distance = sg.Input(key='max_heading_distance', default_text=default_params['max_heading_distance'], size=(10,1), tooltip=tip_distance)

    # Speed
    text_speed_min = sg.Text('Min. Speed [m/s]:', size=(22,1))
    in_speed_min = sg.Input(key='min_speed', default_text=default_params['min_speed'], size=(10,1), tooltip=tip_speed_min)
    text_speed_max = sg.Text('Max. Speed [m/s]:', size=(15,1))
    in_speed_max = sg.Input(key='max_speed', default_text=default_params['max_speed'], size=(10,1), tooltip=tip_speed_max)

    # AOI
    text_aoi = sg.Text('AOI')
    in_aoi = sg.In(size=(80,1), tooltip=tip_aoi)
    browse_aoi = sg.FileBrowse(key='aoi', file_types=(("Shapefile", "*.shp"), (".plan File", "*.plan")), initial_folder=os.path.dirname(default_params['aoi']))

    # Time table
    button_time_table = sg.Button('Edit Table', tooltip=tip_time_table)
    check_time_load = sg.Checkbox('Filter by Time', key='filter_table', default=default_params['filter_table'], tooltip=tip_time_filter)

    # Add to layout
    layout.append([sg.HorizontalSeparator()])
    layout.append([text_filtering])
    layout.append([text_crop, in_crop])
    layout.append([text_depth_mode, depth_mode_combo])
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
    in_x_offset = sg.Input(key='x_offset', default_text=default_params['x_offset'], size=(10,1), tooltip=tip_x_offset)
    
    # Y offset
    text_y_offset = sg.Text('Transducer Offset [Y]:', size=(22,1))
    in_y_offset = sg.Input(key='y_offset', default_text=default_params['y_offset'], size=(10,1), tooltip=tip_y_offset)

    # Add to layout
    layout.append([sg.HorizontalSeparator()])
    layout.append([text_position])
    layout.append([text_x_offset, in_x_offset, sg.VerticalSeparator(), text_y_offset, in_y_offset])


    ##################
    # Detection Params

    text_detect = sg.Text('Detection Parameters\n', font=("Helvetica", 14, "underline"))


    # Model Selection #
    avail_models = get_avail_models()
    avail_models_aliases = list(avail_models.keys())
    model_label = sg.Text("Model Selection:", size=(20, 1), font=("Helvetica", 12), justification="left")
    model_list = sg.Combo(avail_models_aliases, key='rf_model', default_value=default_params['rf_model'], tooltip=tip_model)

    
    # Confidence & IoU #
    conf_label = sg.Text('Score Threshold', size=(20,1))
    conf_slider = sg.Slider((0,1), key='confidence', default_value=default_params['confidence'], resolution=0.05, tick_interval=0.25, orientation='horizontal', tooltip=tip_confidence)
    alpha_label = sg.Text('Alpha Weight', size=(20,1))
    alpha_slider = sg.Slider((0,1), key='alpha', default_value=default_params['alpha'], resolution=0.05, tick_interval=0.25, orientation='horizontal', tooltip=tip_alpha)
    iou_label = sg.Text('IoU Threshold', size=(20,1))
    iou_slider = sg.Slider((0,1), key='iou_threshold', default_value=default_params['iou_threshold'], resolution=0.05, tick_interval=0.25, orientation='horizontal', tooltip=tip_iou)

    # Moving Window #
    check_mov_win = sg.Checkbox('Moving Window', key='moving_window', default=default_params['moving_window'], enable_events=True, tooltip=tip_moving_window)
    if default_params['moving_window'] == True:
        mov_win_status = False
    else:
        mov_win_status = True
    text_mov_win = sg.Text('Window Stride', size=(20,1))
    slide_mov_win = sg.Slider((0,1), key='window_stride', default_value=default_params['window_stride'], resolution=0.025, tick_interval=0.25, orientation='horizontal', disabled=mov_win_status, tooltip=tip_window_stride)

    col_detect_1 = sg.Column([[model_label, model_list], 
                              [check_mov_win],
                              [text_mov_win, slide_mov_win]], 
                              vertical_alignment='top')
    
    col_detect_2 = sg.Column([[conf_label, conf_slider],
                              [alpha_label, alpha_slider],
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
    check_track = sg.Checkbox('Track Objects', key='inference_track', default=default_params['inference_track'], enable_events=True, tooltip=tip_track)
    if default_params['inference_track'] == True:
        track_status = False
    else:
        track_status = True
    text_track_thresh = sg.Text('Tracking Threshold', size=(20,1))
    slide_track = sg.Slider((0,1), key='track_cnt_thresh', default_value=default_params['track_cnt_thresh'], resolution=0.05, tick_interval=0.25, orientation='horizontal', disabled=track_status, tooltip=tip_track_threshold)

    # Add to layout
    layout.append([sg.HorizontalSeparator()])
    layout.append([text_track])
    layout.append([check_track, sg.VerticalSeparator(), text_track_thresh, slide_track])

    #########
    # Exports

    text_exports = sg.Text('Export Options\n', font=("Helvetica", 14, "underline"))

    # Export Image
    check_image = sg.Checkbox('Export Detection Images', key='export_image', default=default_params['export_image'], tooltip=tip_export_image)
    # Export Video
    check_video = sg.Checkbox('Export Detection Videos', key='export_vid', default=default_params['export_vid'], tooltip=tip_export_video)

    # Add to layout
    layout.append([sg.HorizontalSeparator()])
    layout.append([text_exports])
    layout.append([check_image, sg.VerticalSeparator(), check_video])

    #####################
    # Submit/quit buttons
    layout.append([sg.HorizontalSeparator()])
    layout.append([sg.Push(), sg.Submit(), sg.Quit(), sg.Button('Save Defaults'), sg.Push()])

    layout2 =[[sg.Column(layout, scrollable=True,  vertical_scroll_only=True, size_subsample_height=4)]]
    window = sg.Window('GhostVision', layout2, resizable=True, finalize=True)
    _fit_window_to_screen(window)

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

    # Find all supported sonar files in all subdirectories of inDir
    if batch:
        inFiles=[]
        for root, dirs, files in os.walk(inDir):
            if '__MACOSX' not in root:
                for file in files:
                    if file.endswith(SUPPORTED_EXTS):
                        inFiles.append(os.path.join(root, file))

        inFiles = sorted(inFiles)

    else:
        inFiles = [values['inFile']]
    
    
    
    
    # inFiles = inFiles[:1] # for testing




    for i, f in enumerate(inFiles):
        print(i, ":", f)

    # Create output directory if it doesn't exist
    if not os.path.exists(outDir):
        os.makedirs(outDir)

    #============================================
    detect_dep_label = _normalize_depth_mode_value(values.get('detectDep', default_params.get('detectDep', 'Binary Threshold')))
    detect_dep = DEPTH_MODE_OPTIONS[detect_dep_label]

    pm_params = {
        'project_mode': int(values['project_mode']),
        # 'nchunk': int(values['nchunk']),
        'nchunk': 500,
        'cropRange': float(values['cropRange']),
        'threadCnt': 0.5,
        'aoi': aoi,
        'max_heading_deviation': float(values['max_heading_deviation']),
        'max_heading_distance': float(values['max_heading_distance']),
        'min_speed': float(values['min_speed']),
        'max_speed': float(values['max_speed']),
        'time_table': time_table,
        'x_offset': float(values['x_offset']),
        'y_offset': float(values['y_offset']),
        'wcp': True,
        'export_16bit': True,
        'export_colormap_uint8': True,
        'tileFile': '.tif',
        'egn': bool(values.get('egn', True)),
        'egn_stretch': 2,
        'egn_stretch_factor': 0.5,
        'rectMethod': rectMethod,
        'force_rectify': True,
        'side_scan_only': True,
        'detectDep': detect_dep,
    }

    preserve_subdirs = bool(values.get('preserve_subdirs', default_params.get('preserve_subdirs', False))) if batch else False
    log_pm_results = bool(default_params.get('log_pm_results', True))

    pingmapper_doWork = _get_pingmapper_dowork()

    pm_results = pingmapper_doWork(
        in_file=(values['inFile'] if not batch else None),
        in_dir=(inDir if batch else None),
        in_files=(inFiles if batch else None),
        out_dir=outDir,
        proj_name=(values['projName'] if not batch else None),
        prefix=(values['prefix'] if batch else ''),
        suffix=(values['suffix'] if batch else ''),
        batch=batch,
        preserve_subdirs=preserve_subdirs,
        params=pm_params,
        script_path=os.path.abspath(__file__),
    )

    if log_pm_results:
        pm_log_csv = os.path.join(outDir, 'pingmapper_dowork_results.csv')
        with open(pm_log_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['inFile', 'projDir', 'logfilename', 'success'])
            writer.writeheader()
            for run in pm_results:
                writer.writerow({
                    'inFile': run.get('inFile', ''),
                    'projDir': run.get('projDir', ''),
                    'logfilename': run.get('logfilename', ''),
                    'success': run.get('success', False),
                })

    model_alias = values['rf_model']
    rf_model = avail_models[model_alias]

    for run in pm_results:
        if not run.get('success'):
            continue

        projDir = run.get('projDir')
        inFile = run.get('inFile')

        if not projDir or not os.path.exists(projDir):
            continue

        params = dict(pm_params)
        params['projDir'] = projDir
        params['inFile'] = inFile
        params['rf_model'] = rf_model
        params['gpxToHum'] = values['gpxToHum']
        params['sdDir'] = inDir if batch else os.path.dirname(inFile)
        params['confidence'] = values['confidence']
        params['alpha'] = values['alpha']
        params['iou_threshold'] = values['iou_threshold']

        wptPrefix = values['wptPrefix']
        params['wptPrefix'] = wptPrefix
        window_stride = float(values['window_stride'])
        params['stride'] = int(window_stride*nchunk)
        params['moving_window'] = bool(values['moving_window'])
        params['window_stride'] = float(values['window_stride'])

        params['export_vid'] = values['export_vid']
        params['export_image'] = values['export_image']
        params['inference_track'] = values['inference_track']

        tracker_cnt = np.around((nchunk / (window_stride*nchunk)) * values['track_cnt_thresh'], decimals=0)
        if tracker_cnt < 1:
            tracker_cnt = 1
        params['tracker_cnt'] = ((nchunk / (window_stride*nchunk)) * values['track_cnt_thresh'])

        if params['export_vid'] and not params['export_image']:
            params['export_image'] = True
            params['delete_image'] = True

        # Normalize WCP tiles for XTF/JSF before detection
        xtf_jsf = inFile.lower().endswith(('.xtf', '.jsf'))
        if xtf_jsf:
            wcp_tiles = glob(os.path.join(projDir, '**', 'wcp_*.tif'), recursive=True)
            if wcp_tiles:
                _normalize_wcp_tiles(wcp_tiles)

        print('\n\n', '***User Detection Parameters***')
        for k,v in params.items():
            print("| {:<20s} : {:<10s} |".format(k, str(v)))

        print('\n===========================================')
        print('===========================================')
        print('***** DETECTING CRAB POTS *****')
        crabpots_master_func(**params)

    print('\n===========================================')
    print('===========================================')
    print('***** EXPORTING FINAL RESULTS *****')
    export_final_results(outDir, os.path.basename(outDir))

    print("\n\nGrand Total Processing Time: ",datetime.timedelta(seconds = round(time.time() - start_time, ndigits=0)))

def get_avail_models():

    # Get all available models from GitHub
    if not os.path.exists(rf_model_dir):
        os.makedirs(rf_model_dir)

    # download_all_models(rf_model_dir)

    # # Get projects in directory
    # projects = os.listdir(rf_model_dir)

    
    # # Find all folders and subfolders in rf_model_dir
    # avail_models = []
    # for proj in projects:
    #     versions = os.listdir(os.path.join(rf_model_dir, proj))

    #     for v in versions:
    #         avail_models.append('{}/{}'.format(proj, v))

    avail_models = download_all_models(rf_model_dir)

    return avail_models

def download_all_models(rf_model_dir):

    avail_models = {}

    # Known models
    known_models = {
        'ghostvision-models/1': ['ghostvision_rf-detr-v1.zip', 'rf-detr_v1'],
        'ghostvision-models/2': ['ghostvision_yolo26-v1.zip', 'yolo26_v1'],
        'ghostvision-models/5': ['ghostvision_yolo12-v1.zip', 'yolo12_v1'],
    }
    
    url = r'https://github.com/PINGEcosystem/GhostVision/releases/download/models'

    for k, v in known_models.items():
        model_dir = os.path.join(rf_model_dir, k)
        model_zip_name = v[0]
        model_alias = v[1]
        if os.path.exists(model_dir):
            avail_models[model_alias] = k
            continue

        print('Downloading model: {}'.format(k))
        model_url = '{}/{}'.format(url, model_zip_name)
        zip_path = os.path.join(rf_model_dir, model_zip_name)

        try:
            response = requests.get(model_url, stream=True, timeout=120)
            response.raise_for_status()

            with open(zip_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)

            if not zipfile.is_zipfile(zip_path):
                content_type = response.headers.get('Content-Type', 'unknown')
                raise zipfile.BadZipFile(
                    'Downloaded asset is not a zip file for {} from {} (content-type: {})'.format(
                        k,
                        model_url,
                        content_type,
                    )
                )

            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(rf_model_dir)

            print('Model downloaded and extracted to: {}'.format(model_dir))
            avail_models[model_alias] = k

        except (requests.RequestException, zipfile.BadZipFile) as exc:
            print('WARNING: Could not download model {}: {}'.format(k, exc))
            print('Skipping unavailable model asset: {}'.format(model_zip_name))

        finally:
            if os.path.exists(zip_path):
                os.remove(zip_path)

    return avail_models


if __name__ == "__main__":
    detect_main()