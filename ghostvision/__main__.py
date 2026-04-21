'''
Copyright (c) 2025 Cameron S. Bodine
'''


import os, sys

import FreeSimpleGUI as sg

# Debug
detectPath = os.path.normpath('../PINGDetect')
detectPath = os.path.abspath(detectPath)
sys.path.insert(0, detectPath)

# Add 'ghostvision' to the path, may not need after pypi package...
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PACKAGE_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.append(PACKAGE_DIR)

# Set GHOSTVISION utils dir
USER_DIR = os.path.expanduser('~')
GV_UTILS_DIR = os.path.join(USER_DIR, '.ghostvision')
if not os.path.exists(GV_UTILS_DIR):
    os.makedirs(GV_UTILS_DIR)

def main(process):

    from ghostvision.detect import detect_main
    detect_main()

if __name__ == "__main__":
    
    if len(sys.argv) <= 1:
        to_do = 'detect'
        main(to_do)

    elif sys.argv[1] == "rf-download":
        from pingdetect.rf_utils import get_model

        get_model(GV_UTILS_DIR)

    # elif sys.argv[1] == "folder":
    #     from ghostvision.inference_folder import do_inference

    #     dir = sys.argv[2]

    #     do_inference(dir)
