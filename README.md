# GhostVision

# 🚧**UNDER CONSTRUCTION**🚧

[![PyPI - Version](https://img.shields.io/pypi/v/ghostvision?style=flat-square&label=Latest%20Version%20(PyPi))](https://pypi.org/project/ghostvision/)
[![GitHub last commit](https://img.shields.io/github/last-commit/PINGEcosystem/GhostVision)](https://github.com/PINGEcosystem/GhostVision/commits)
[![GitHub commit activity](https://img.shields.io/github/commit-activity/m/PINGEcosystem/GhostVision)](https://github.com/PINGEcosystem/GhostVision/commits)
[![GitHub](https://img.shields.io/github/license/PINGEcosystem/GhostVision)](https://github.com/PINGEcosystem/GhostVision/blob/main/LICENSE)

Near-real time detection of derelict (ghost) crab pots with side-scan sonar.

![ezgif com-crop](https://github.com/user-attachments/assets/ece0602b-1edf-4a2a-88ec-9301b2483378)

## Overview

`GhostVision` is an open-source Python interface for automatically detecting and mapping ghost (derelict) crab pots from side-scan sonar imagery. `GhostVision` leverages [`Yolo`](https://docs.ultralytics.com/) models trained with [`Roboflow`](https://roboflow.com/). Detections are then georeferenced with [`PINGMapper`](https://github.com/CameronBodine/PINGMapper).

## Installation

### GPU (Fast Inference)

`GhostVision` is optimized for running inference (predictions) on the GPU. The processing environment is installed with `conda`. Any flavor of `conda` will do, but we recommend [`Miniforge`](https://conda-forge.org/download/). Follow the instructions below based on your OS.

#### Windows Only
Windows does not natively support inference on the GPU. A utility called [WSL](https://learn.microsoft.com/en-us/windows/wsl/install) (Windows Subsystem for Linux) needs to be installed in order to run inference on the GPU.

1. Install [WSL](https://learn.microsoft.com/en-us/windows/wsl/install) (Windows Subsystem for Linux) & 
2. Open the command prompt by launching `Ubuntu` from the Windows Start menu.

#### Install `Miniforge`

3. In a command prompt, download [`Miniforge`](https://conda-forge.org/download/) with:
    ```
    wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
    ```
4. Install [`Miniforge`](https://conda-forge.org/download/) with:
    ```
    bash Miniforge3-$(uname)-$(uname -m).sh
    ```

#### Install `GhostVision`

5. Install `PINGInstaller`:
    ```
    pip install pinginstaller
    ```
6. Install `GhostVision`:
    ```
    python -m pinginstaller ghostvision-gpu
    ```

### CPU (Slow Inference; Experimental)
An experimental version of `GhostVision` is available to test inference speeds on the CPU. This has been tested on Windows 11 only.

1. Install [`Miniforge`](https://conda-forge.org/download/).
2. Open the [`Miniforge`](https://conda-forge.org/download/) prompt.
3. Install `PINGInstaller`:
    ```
    pip install pinginstaller
    ```
4. Install `GhostVision`.
    ```
    python -m pinginstaller ghostvision
    ```

## Usage

1. Open the appropriate command prompt based on your installation above.
2. Launch `GhostVision`:
    ```
    conda activate ghostvision
    python -m ghostvision
    ```
3. Select desired parameters and click `Submit`.

## Download Custom `Roboflow` Object Detection Model

`GhostVision` includes `Roboflow` object detection models designed to detect crab pots from side-scan sonar imagery. You can train and use your own object detection model by downloading the model from `Roboflow` with the included utility.

1. Open the appropriate command prompt based on your installation above.
2. Launch the Roboflow model download utility:
    ```
    conda activate ghostvision
    python -m ghostvision rf-download
    ```
3. Supply your [Roboflow API Key](https://docs.roboflow.com/developer/authentication/find-your-roboflow-api-key).
4. Enter the project name (all lowercase).
5. Enter the project version.

The model will be downloaded and available to use.

## Acknowledgments

`GhostVision` has been made possible through mentorship, partnerships, financial support, open-source software, manuscripts, and documentation linked below.

*NOTE: The contents of this repository are those of the author(s) and do not necessarily represent the views of the individuals and organizations specifically mentioned here.*

**Development Team:** [Cameron Bodine](https://github.com/CameronBodine), [Art Trembanis](https://www.udel.edu/academics/colleges/ceoe/departments/smsp/faculty/arthur-trembanis/), Kleio Baxevani, Naveed Abbasi, Onur Bagoren, Olivia Hines, Jared Wierzbicki, Ophelia Christoph, Catherine Hughes, Julia Greco.

- [Coastal Sediments, Hydrodynamics and Engineering Lab (CSHEL)](https://sites.udel.edu/ceoe-art/), [College of Earth, Ocean, & Environment (CEOE)](https://www.udel.edu/ceoe/), [University of Delaware](https://www.udel.edu/)

- [Project ABLE (Align, Build Leverage, and Expand)](https://sites.udel.edu/ceoe-able/)

- [National Fish Trap, Removal, Assessent, and Prevention (TRAP) Program](https://trapprogram.org/)

- [Delaware Sea Grant](https://www.udel.edu/academics/colleges/ceoe/delaware-sea-grant/)
