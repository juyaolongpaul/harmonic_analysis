---
layout: default
title: Installation Guide

---

## Installation Guide

1. Use `git clone git@github.com:juyaolongpaul/harmonic_analysis.git` in the terminal to clone the project, then use `cd harmonic_analysis` to go into the project folder. Making sure that you are on the master branch.
2. Create a virtual environment using Python 3. Making sure you have it already installed in your computer, and added to `PATH`. Also, making sure you have `virtualenv` already installed. To install it in Windows, please refer [here](https://programwithus.com/learn-to-code/Pip-and-virtualenv-on-Windows/), until "Launch virtualenv" Section since you are going to use the following commands to launch it. In Linux, an example is: `virtualenv .env --python=python3.5`. Please change `python3.5` into the one installed in your machine. For example, if your machine has Python 3.6, then use `virtualenv .env --python=python3.6`. In Windows, use `virtualenv .env --python=python`, and in Mac OS, use `virtualenv .env --python=python3`.
3. Activate the virtual environment. If you use the command line provided in the second step, you can activate it by `source ./.env/bin/activate` in Mac OS and Linux; in Windows, it is `source .\.env\Scripts\activate`.
4. Use `pip install -r requirements_gpu.txt` to install the required packages if you have a CUDA-compatiable GPU (you need to download [CUDA](https://developer.nvidia.com/cuda-90-download-archive) and [cuDNN]( https://developer.nvidia.com/cudnn) first and install them) and you want to train the networks on GPU; use `pip install -r requirements_cpu.txt` if you want to train the networks on CPU.
5. Use `python main.py` to run the project with the default parameter, where you can visit [here](https://juyaolongpaul.github.io/harmonic_analysis/Github_Page/parameter_adjustment.html) to see the details. 
