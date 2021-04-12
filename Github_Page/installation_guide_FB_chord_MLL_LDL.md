---
layout: default
title: Installation Guide

---

## Guide
This guide is to deploy the multi-label learning and label distritbuion learning of automatic chord labelling work introduced in Yaolong Ju's dissertation "Addressing ambiguity in supervised machine learning: A case study on automatic chord labelling", Section 7.2 and Section 7.3.
1. Use `git clone --recurse-submodules git@github.com:juyaolongpaul/harmonic_analysis.git` in the terminal to clone the project, then use `cd harmonic_analysis` to go into the project folder. Making sure that you are on the `multi-label` branch (using `git checkout multi-label` to switch).
2. Create a virtual environment using Python 3. Making sure you have it already installed in your computer, and added to `PATH`. Also, making sure you have `virtualenv` already installed. To install it in Windows, please refer [here](https://programwithus.com/learn-to-code/Pip-and-virtualenv-on-Windows/), until "Launch virtualenv" Section since you are going to use the following commands to launch it. In Linux and Mac OS, an example is: `virtualenv .env --python=python3`. In Windows, use `virtualenv .env --python=python`.
3. Activate the virtual environment. If you use the command line provided in the second step, you can activate it by `source ./.env/bin/activate` in Mac OS and Linux; in Windows, it is `.\.env\Scripts\activate`.
4. Use `pip install -r requirements_cpu.txt` to install the required packages.
5. Use `python FB2lyrics.py` to run the script to generate all the necessary file for multi-label learning and label distribution learning.
6. Use `python -s MLL_BCMCL main_FB.py` to run the project for multi-label learning (proposed in Section 7.2). Use `python -s LDL_BCMCL main_FB.py` to run the project for label distribution learning (proposed in Section 7.3). 

If you have questions or further inquiries, please send an email to `yaolong.ju@mail.mcgill.ca`. 

 
