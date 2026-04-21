# GhostVision

# 🚧**UNDER CONSTRUCTION**🚧

[![PyPI - Version](https://img.shields.io/pypi/v/ghostvision?style=flat-square&label=Latest%20Version%20(PyPi))](https://pypi.org/project/ghostvision/)
[![GitHub last commit](https://img.shields.io/github/last-commit/PINGEcosystem/GhostVision)](https://github.com/PINGEcosystem/GhostVision/commits)
[![GitHub commit activity](https://img.shields.io/github/commit-activity/m/PINGEcosystem/GhostVision)](https://github.com/PINGEcosystem/GhostVision/commits)
[![GitHub](https://img.shields.io/github/license/PINGEcosystem/GhostVision)](https://github.com/PINGEcosystem/GhostVision/blob/main/LICENSE)

Near-real time detection of derelict (ghost) crab pots with side-scan sonar.

![ezgif com-crop](https://github.com/user-attachments/assets/ece0602b-1edf-4a2a-88ec-9301b2483378)

## 📡 Overview

`GhostVision` is an open-source Python interface for automatically detecting and mapping ghost (derelict) crab pots from side-scan sonar imagery. `GhostVision` currently supports multiple packaged object-detection models, including YOLO- and RF-DETR-based exports trained with [`Roboflow`](https://roboflow.com/). Detections are then georeferenced with [`PINGMapper`](https://github.com/CameronBodine/PINGMapper).

## 📜 Published Documentation
### 📄 Journal Article
Bodine, C.S.; Baxevani, K.; Abbasi, N.;Wierzbicki, J.; Christoph, O.; Hughes, C.; Bagoren, O.; Hines, O.; Greco, J.; Trembanis, A. GhostVision: Democratizing Derelict Gear Detection using Low-Cost Sonar and Artificial Intelligence. (In Review). Submitted to Journal of Marine Science and Engineering.

### 🤖 Models
- GV-RF-DETR [![Model on HF](https://huggingface.co/datasets/huggingface/badges/resolve/main/model-on-hf-md.svg)](https://huggingface.co/PINGEcosystem/gv-rf-detr)
- GV-YOLO12 [![Model on HF](https://huggingface.co/datasets/huggingface/badges/resolve/main/model-on-hf-md.svg)](https://huggingface.co/PINGEcosystem/gv-yolo12)
- GV-YOLO26 [![Model on HF](https://huggingface.co/datasets/huggingface/badges/resolve/main/model-on-hf-md.svg)](https://huggingface.co/PINGEcosystem/gv-yolo26)

### 🗂️ Model Dataset
[![Dataset on HF](https://huggingface.co/datasets/huggingface/badges/resolve/main/dataset-on-hf-md.svg)](https://huggingface.co/datasets/PINGEcosystem/sss-crab-pot-detection-ds)

## ⚙️ Installation

### 🚀 GPU (Fast Inference)

`GhostVision` is optimized for running inference (predictions) on the GPU. The processing environment is installed with `conda`. Any flavor of `conda` will do, but we recommend [`Miniforge`](https://conda-forge.org/download/). Follow the instructions below based on your OS.

#### Windows Only
Windows does not natively support inference on the GPU. A utility called [WSL](https://learn.microsoft.com/en-us/windows/wsl/install) (Windows Subsystem for Linux) needs to be installed in order to run inference on the GPU.

1. Install the [latest NVIDIA driver](https://docs.nvidia.com/cuda/wsl-user-guide/index.html#getting-started-with-cuda-on-wsl-2) for your system.
2. Add [CUDA Support for WSL 2](https://docs.nvidia.com/cuda/wsl-user-guide/index.html#cuda-support-for-wsl-2).
    - *Assumes your computer has an NVIDIA GPU.*
3. Install [WSL](https://learn.microsoft.com/en-us/windows/wsl/install) (Windows Subsystem for Linux) & 
4. Open the command prompt by launching `Ubuntu` from the Windows Start menu.
5. You may need to install the NVIDIA Cuda Toolkit with `sudo apt install nvidia-cuda-toolkit`.

#### Install `Miniforge`

4. In a command prompt, download [`Miniforge`](https://conda-forge.org/download/) with:
    ```
    wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
    ```
5. Install [`Miniforge`](https://conda-forge.org/download/) with:
    ```
    bash Miniforge3-$(uname)-$(uname -m).sh
    ```

#### Install `GhostVision`

6. Install `PINGInstaller`:
    ```
    pip install pinginstaller
    ```
7. Install `GhostVision`:
    ```
    python -m pinginstaller ghostvision-gpu
    ```

### 🐢 CPU (Slow Inference; Experimental)
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

## 🚀 Usage

1. Open the appropriate command prompt based on your installation above.
2. Launch `GhostVision`:
    ```
    conda activate ghostvision
    python -m ghostvision
    ```
3. Select your desired model and processing parameters, then click `Submit`.

Bundled release models are downloaded automatically into the local `~/.ghostvision/models` cache the first time they are needed. The current packaged aliases exposed by the app include `rf-detr_v1`, `yolo26_v1`, and `yolo12_v1`.

## Recommended Settings

The most useful model-specific operating points currently come from the full GhostVision accuracy assessment workflow used for the companion manuscript. In that analysis, detections were evaluated against manual annotations at a `3 m` match radius. When object tracking is enabled, GhostVision combines confidence and temporal persistence using:

`S = alpha * conf_avg + (1 - alpha) * pred_cnt / max(pred_cnt)`

The combined-score analysis reports the following best-performing `alpha` values:

- `YOLOv12`: `alpha = 0.90`
- `YOLOv26`: `alpha = 0.95`
- `RF-DETR`: `alpha = 0.85`

The same workflow reports these best thresholds:

- `YOLOv12`: `confidence = 0.148`, `pred_cnt = 17`, `combined score = 0.221`
- `YOLOv26`: `confidence = 0.101`, `pred_cnt = 15`, `combined score = 0.136`
- `RF-DETR`: `confidence = 0.386`, `pred_cnt = 18`, `combined score = 0.345`

Peak F1 values from that workflow were:

- `YOLOv12`: `F1 = 0.739` from confidence alone, `0.574` from persistence alone, `0.716` from the combined score
- `YOLOv26`: `F1 = 0.737` from confidence alone, `0.596` from persistence alone, `0.707` from the combined score
- `RF-DETR`: `F1 = 0.721` from confidence alone, `0.667` from persistence alone, `0.727` from the combined score

For practical use, this suggests:

- Prefer `YOLOv12` as the default packaged model for the best overall operational balance.
- Start `YOLOv12` near `score threshold = 0.15`, `pred_cnt = 17`, and `alpha = 0.90` when you want settings that reproduce the evaluation workflow.
- Use `RF-DETR` only when very high recall is more important than false-positive burden.

GhostVision now uses the same `pred_cnt / max(pred_cnt)` normalization as the evaluation workflow when computing the tracked combined score. These values should be treated as model-specific reference points, not universal defaults. GhostVision's packaged defaults remain general-purpose starting values for field use, while the values above are the best choices when you want to reproduce the evaluation workflow as closely as possible.

## 📦 Download Custom `Roboflow` Object Detection Model

`GhostVision` includes packaged object detection models designed to detect crab pots from side-scan sonar imagery. If you want to use your own compatible `Roboflow` export instead, you can download a custom model with the included utility.

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

## 🔗 Related Resources

- Dataset: [PINGEcosystem/sss-crab-pot-detection-ds](https://huggingface.co/datasets/PINGEcosystem/sss-crab-pot-detection-ds)
- RF-DETR model card: [PINGEcosystem/gv-rf-detr](https://huggingface.co/PINGEcosystem/gv-rf-detr)
- YOLO12 model card: [PINGEcosystem/gv-yolo12](https://huggingface.co/PINGEcosystem/gv-yolo12)
- YOLO26 model card: [PINGEcosystem/gv-yolo26](https://huggingface.co/PINGEcosystem/gv-yolo26)

## 🙌 Acknowledgments

`GhostVision` has been made possible through mentorship, partnerships, financial support, open-source software, manuscripts, and documentation linked below.

*NOTE: The contents of this repository are those of the author(s) and do not necessarily represent the views of the individuals and organizations specifically mentioned here.*

**Development Team:** [Cameron Bodine](https://github.com/CameronBodine), [Art Trembanis](https://www.udel.edu/academics/colleges/ceoe/departments/smsp/faculty/arthur-trembanis/), Kleio Baxevani, Naveed Abbasi, Onur Bagoren, Olivia Hines, Jared Wierzbicki, Ophelia Christoph, Catherine Hughes, Julia Greco.

- [Coastal Sediments, Hydrodynamics and Engineering Lab (CSHEL)](https://sites.udel.edu/ceoe-art/), [College of Earth, Ocean, & Environment (CEOE)](https://www.udel.edu/ceoe/), [University of Delaware](https://www.udel.edu/)

- [Project ABLE (Align, Build Leverage, and Expand)](https://sites.udel.edu/ceoe-able/)

- [National Fish Trap, Removal, Assessent, and Prevention (TRAP) Program](https://trapprogram.org/)

- [Delaware Sea Grant](https://www.udel.edu/academics/colleges/ceoe/delaware-sea-grant/)
