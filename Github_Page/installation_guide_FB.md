---
layout: default
title: Installation Guide

---

## Guide

1. Use `git clone git@github.com:juyaolongpaul/harmonic_analysis.git` in the terminal to clone the project, then use `cd harmonic_analysis` to go into the project folder. Making sure that you are on the master branch.
2. Create a virtual environment using Python 3. Making sure you have it already installed in your computer, and added to `PATH`. Also, making sure you have `virtualenv` already installed. To install it in Windows, please refer [here](https://programwithus.com/learn-to-code/Pip-and-virtualenv-on-Windows/), until "Launch virtualenv" Section since you are going to use the following commands to launch it. In Linux and Mac OS, an example is: `virtualenv .env --python=python3`. In Windows, use `virtualenv .env --python=python`.
3. Activate the virtual environment. If you use the command line provided in the second step, you can activate it by `source ./.env/bin/activate` in Mac OS and Linux; in Windows, it is `.\.env\Scripts\activate`.
4. Use `pip install -r requirements_gpu.txt` to install the required packages if you have a CUDA-compatiable GPU (you need to download [CUDA](https://developer.nvidia.com/cuda-90-download-archive) and [cuDNN]( https://developer.nvidia.com/cudnn) first and install them) and you want to train the networks on GPU; use `pip install -r requirements_cpu.txt` if you want to train the networks on CPU. Please make sure your python version is 3.7 or under! Currently, python 3.8 does not work because tensorflow cannot be installed.
5. Use `python main_FB.py` to run the project with the default parameter, and you should be able to achieve the same results (i.e., the result of DNN) reported in the [paper](). The generate figured bass (saved as MusicXML files) can be found in the directory:`./predicted_result/Bach_o_FB/NCT_pitch_classpitch_class3meter_NewOnsetDNN1_2_rule_3/`.One generated result is shown below: ![image](https://user-images.githubusercontent.com/9313094/89348526-b5a40180-d67a-11ea-9e16-bf137b4d2f45.png) One only needs to observe the last two voices (at the bottom), where the last voice contains the generated figured bass annotations (FBAs) by the rule-based algorithm, and the second last voice contains both ground truth FBAs (upper), and the generated FBAs by machine learning algorithms (lower). Underneath the generated FBAs, there are results, where “✓” means that the generated figured bass exactly matches Bach’s FBAs (the ground truth), “✓_” means they are considered correct by our evaluation metric that treats musically equivalent figures as equivalent (see the [paper]() of Section 3.3.2). “✘” means the generated figures are considered as errors. 

If you have questions or further inquiries, please send an email to `yaolong.ju@mail.mcgill.ca`. 

 
