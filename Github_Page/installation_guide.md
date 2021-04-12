---
layout: default
title: Installation Guide

---

## Installation Guide

1. Download and unzip this project: https://github.com/juyaolongpaul/harmonic_analysis/releases/tag/v1.0, then use `cd harmonic_analysis` to go into the project folder. Making sure that you are on the master branch.
2. Create a virtual environment using Python 3. Making sure you have it already installed in your computer, and added to `PATH`. Also, making sure you have `virtualenv` already installed. To install it in Windows, please refer [here](https://programwithus.com/learn-to-code/Pip-and-virtualenv-on-Windows/), until "Launch virtualenv" Section since you are going to use the following commands to launch it. In Linux and Mac OS, an example is: `virtualenv .env --python=python3`. In Windows, use `virtualenv .env --python=python`.
3. Activate the virtual environment. If you use the command line provided in the second step, you can activate it by `source ./.env/bin/activate` in Mac OS and Linux; in Windows, it is `.\.env\Scripts\activate`.
4. Use `pip install -r requirements_gpu.txt` to install the required packages if you have a CUDA-compatiable GPU (you need to download [CUDA](https://developer.nvidia.com/cuda-90-download-archive) and [cuDNN]( https://developer.nvidia.com/cudnn) first and install them) and you want to train the networks on GPU; use `pip install -r requirements_cpu.txt` if you want to train the networks on CPU. Please make sure your python version is 3.7 or under! Currently, python 3.8 does not work because tensorflow cannot be installed.
5. Use `python main.py` to run the project with the default parameter, where you can visit [here](https://juyaolongpaul.github.io/harmonic_analysis/Github_Page/parameter_adjustment.html) to see the details. 
