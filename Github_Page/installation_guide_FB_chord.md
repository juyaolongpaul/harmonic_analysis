---
layout: default
title: Installation Guide

---

## Guide
The chord labels can already be found at the Bach Chorales Multiple Chord Labels ([BCMCL](https://github.com/juyaolongpaul/BCMCL)) repo. If you want to actually run the script, you need to follow the steps below. Note that the generated results will overwrite the ones you cloned from the BCMCL repo.
1. Use `git clone --recurse-submodules git@github.com:juyaolongpaul/harmonic_analysis.git` in the terminal to clone the project, then use `cd harmonic_analysis` to go into the project folder. Making sure that you are on the master branch.
2. Create a virtual environment using Python 3. Making sure you have it already installed in your computer, and added to `PATH`. Also, making sure you have `virtualenv` already installed. To install it in Windows, please refer [here](https://programwithus.com/learn-to-code/Pip-and-virtualenv-on-Windows/), until "Launch virtualenv" Section since you are going to use the following commands to launch it. In Linux and Mac OS, an example is: `virtualenv .env --python=python3`. In Windows, use `virtualenv .env --python=python`.
3. Activate the virtual environment. If you use the command line provided in the second step, you can activate it by `source ./.env/bin/activate` in Mac OS and Linux; in Windows, it is `.\.env\Scripts\activate`.
4. Use `pip install -r requirements_cpu.txt` to install the required packages.
5. Use `python FB2lyrics.py` to run the script. The generate figured bass (saved as MusicXML files) can be found in the directory:`./Bach_chorale_FB/FB_source/musicXML_master/BCMCL/`. To see the example of the generated chord labels, please see the documentation [here](https://github.com/juyaolongpaul/BCMCL).

If you have questions or further inquiries, please send an email to `yaolong.ju@mail.mcgill.ca`. 

 
