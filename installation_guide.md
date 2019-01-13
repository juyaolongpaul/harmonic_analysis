---
layout: default
title: Installation Guide

---

## Installation Guide

1. Use `git clone git@github.com:juyaolongpaul/harmonic_analysis.git` in the terminal to clone the project, then use `cd harmonic_analysis` to go into the project folder. Then, change the branch into `non-chord-tone` by `git checkout non-chord-tone`.
2. Create a virtual environment using Python 3. An example is: `virtualenv .env --python=python3.5`. Please change `python3.5` into the one installed in your machine. For example, if your machine has Python 3.6, then use `virtualenv .env --python=python3.6`. In Windows, if this command does not work, try `virtualenv .env --python=python` instead.
3. Activate the virtual environment. If you use the command line provided in the second step, you can activate it by `source ./.env/bin/activate` in Mac OS and Linux; in Windows, it is ` .\.env\Scripts\activate`.
4. Use `pip install -r requirements_gpu.txt` to install the required packages if you have a CUDA-compatiable GPU (you need to download [CUDA](https://developer.nvidia.com/cuda-90-download-archive) first and install it) and you want to train the networks on GPU; use `pip install -r requirements_cpu.txt` if you want to train the networks on CPU.
5. Use `python main.py` to run the project.
