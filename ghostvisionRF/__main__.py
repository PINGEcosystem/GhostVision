'''
Copyright (c) 2025 Cameron S. Bodine
'''


import os, sys

# Add 'ghostvision' to the path, may not need after pypi package...
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PACKAGE_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.append(PACKAGE_DIR)

def main(process):

    from ghostvisionRF.detect import detect_main
    detect_main()




if __name__ == "__main__":
    if len(sys.argv) == 0:
        to_do = 'detect'
        main(to_do)
    
    if len(sys.argv) == 1:
        to_do = 'detect'
        main(to_do)

    elif sys.argv[1] == "roboflow":
        from ghostvisionRF.download_roboflow import get_model

        get_model()

    elif sys.argv[1] == "folder":
        from ghostvisionRF.inference_folder import do_inference

        dir = sys.argv[2]

        do_inference(dir)
