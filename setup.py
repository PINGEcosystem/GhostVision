from setuptools import setup, find_packages
from pathlib import Path

DESCRIPTION = 'Near-real time detection of derelict (ghost) crab pots with side-scan sonar.'
LONG_DESCRIPTION = Path('README.md').read_text()

exec(open('ghostvision/version.py').read())

setup(
    name="ghostvision",
    version=__version__,
    author="Cameron Bodine",
    author_email="bodine.cs@gmail.email",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type='text/markdown',
    packages=find_packages(),
    data_files=[("ghostvision_config", ["ghostvision/default_params.json"])],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Visualization",
        "Topic :: Scientific/Engineering :: Oceanography",
        "Topic :: Scientific/Engineering :: GIS",
        "Topic :: Scientific/Engineering :: Hydrology"
        ],
    keywords=[
        "pingmapper",
        "sonar",
        "ecology",
        "remotesensing",
        "sidescan",
        "sidescan-sonar",
        "aquatic",
        "humminbird",
        "lowrance",
        "gis",
        "oceanography",
        "limnology",
        "object-detection",],
    python_requires=">=3.6",
    install_requires=['pinginstaller', 'pingwizard', 'pingverter', 'pingmapper', 'pingdetect'],
    project_urls={
        "Issues": "https://github.com/PINGEcosystem/GhostVision/issues",
        "GitHub":"https://github.com/PINGEcosystem/GhostVision",
        # "Homepage":"https://cameronbodine.github.io/PINGMapper/",
    },
)