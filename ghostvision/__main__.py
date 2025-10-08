'''
Copyright (c) 2025 Cameron S. Bodine
'''


import os, sys

PySimpleGUI_License = 'e3yAJ9MVaOWANplCbmndNNl2VwHvlCwpZTSjIl6DIjkiRGpYc53aRty8aBWpJF1qdwGLlzv9bUiHILs3Inkyxpp5Yq2OVku8cg2ZVrJ7RNCQI66bMcTLcnyKMbTRMK57OCTPMGxGNtS8whirTBGTlLjxZEWg5DzWZdUXRUlLcDGfxnv7eiWB1jlOb6nqR8WTZ2XsJVzbabW19ouWI6j0oXiKN0Si4AwtI7iFw8iGTBmtFftjZEUxZMpYcLncNk0rIJj4oyisQq2uFCtqZnXWJvvqbEiCICsSIbkC5jhKbvWTVqM2YtX6Ni0XIJjloji1QEmU9Ak5ayWp5nlnIwi3wiiOQK279ytqcKGwFGuvepS6IH6iIOiYIGs7I4kYNQ13cY33RkvIbqWkVyypSKUOQoiZO2ijIFzaMNTEAp0bNxyWI1sLIwkRRjhZdNGBVoJkcZ3MNN1yZMWTQtihOiieIYyXMDDIIF0ILaTyAt36LKTREj5JI1iYwcixRgGuFk0BZGU5VZ4dciGUl3ykZ3XtMbilOMiBIhy1M5DtId1mL6T1A935LYTLEN5iI3iJwoirR8Wa12h5a0WtxkBNZdGiRJyYZXX9N5zZI2jSoZizYpmp9YkHaIWz5YluLTmcNXzNQmGZd0twYGW6l3sALZmTNWvubcSEItsPITk6lFQgQUWZRrkfcEmAVxz0c9y7IG6sILjZEYyzO8Cf4c0WLDj3QCwSLwjPEt2BMMi0J69p854e39898f71ea82d3a530f7a6ed8a02a4eea9ffd2c7b1279074b491c71b411f392e6d726a2d2f9dbf63388356cf4e083e358fe428852d676073e128607b9ad194c15e34a4feb463a749fd3295606caa293b823d102e854cd845b79b5ec5eaec0b2ef7f9cf0c87b2dfcad3f14cd0d66a2da97e6b38a535eb8707b4486c9802a4bfeb09703382e157449096f0e3551af9f444197cacb3f3d42187cea97ab61978985ddeecd086b9cb86c4ec1c08082d47b3ed0ae9c044d9aa65e5c9bf6e00238f78ed858cfdaf0021fb95d636e0cce84d84d2c2da7ac57f2e54fe793fce44a8b8abf96ce7c381f4b7eeb55dc4b68768e8172a4dffc1b683e62a108b2dfc2ef340dab058e6ee5c1f525f93e89d39258862f099987a8ec7022db5aecb5a58e81d02370d5717d18498ae58749aa5e463cf757ab7fa84efe49c1b770da397eef22423696ad433e7232646e279906bef084b21714ac5fc2af564a03ebc789123aed44531765b3e72c6165131feab68e35e0276a64760ee9abf043bece1e3cd148bcec97ab835395391387ff9d2b74a835a15ea5bac9c7e1218c217481a3999a91e037a138aaf5dddadb2247141242140b130e273aab5e1e6855fae8b7ee80d64be2d09a46f3d49555f53a7a849138fc3b9d2323658ea7e86a0039c40f3c15fd3647f99ec98232d9734a5933177c48c6575a1415e2808640cfb27773e728fe128b99757'
import PySimpleGUI as sg

# # Debug
# detectPath = os.path.normpath('../PINGDetect')
# detectPath = os.path.abspath(detectPath)
# sys.path.insert(0, detectPath)

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
