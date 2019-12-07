---
layout: default
title: Parameter Adjustment

---

## Parameter Adjustment

In this program, there are many adjustable parameters, ranging from the analytical styles, types of machine learning models, model architectures, and hyper-parameters. All of the parameters and the corresponding available values are shown in the chart below. The first value will be the default one. If you look at the `main.py` script, you will notice there are a few more parameters. However, they are either not essential for the task or not fully tested yet. Please let me know if you want to experiment with these parameters. 

Parameters   |Values   | Explanation
---|---|---
--source (-s)   |`ISMIR2019`   |The kind of annotations you want to use for training. Currently, only `ISMIR2019` went through the whole workflow, including partial manual modification and re-training. Currently, the code only accepts `ISMIR2019`.
--num_of_hidden_layer (-l)   |3, usually ranging from 2-5   |The number of hiddel layers (not effective in `SVM`)
--num_of_hidden_node (-n)   |300, usually ranging from 100-500  |The number of hidden nodes (not effective in `SVM`)
--model (-m)   |`DNN` is default, `SVM`   also available |The types of models you want to use
--pitch (-p)   |`pitch_class`  |The kind of pitch you want to use as features. `pitch_class` means using 12-d pitch class for each sonority; `pitch_class_4_voices` means using 12-d pitch class for each of the 4 voices
--window (-w)  |1, usually ranging from 0-5|The static window you can add as context for `DNN` or `SVM` model   
--output (-o)   |`NCT_pitch_class`| `NCT_pitch_class` means using 12-d output vector specifying which pitch classes contain non-chord tones (NCTs)
--input (-i)   |`3meter`, `barebone`, `2meter` and `NewOnset` also available. The default is `3meter_NewOnset`.  | Specify what input features, besides pitch classes, you are using. You can use meter features: `2meter` means you are using on/off beat feature; `3meter` means you are using 'strong beat, on/off beat' feature; `NewOnset` means whether the current slice has a real attack or not across all the pitch classes/voices. It will add another 12-d vector in the input specifying which pitch classes are the real attacks. 
--predict (-pre)   |'Y' is the default, 'N' also available|Specify whether you want to predict and output the results in musicXML
--cross_validation (-c)   | 1, ranging from 1-10|Specify how many cross validation folds you want to use. The default is set to 1 for the fast runtime of the experiment
 

### Usage Example
* If you just want to replicate the results from the paper, you can simply use all the default parameters, and the command is `python main.py` This default setting only runs the first fold of cross validation. To run the full experiment, run `python main.py -c 10`. Note that this will give you the re-trained result directly. 
* If you want to change the parameters, say you want to use DNN with 5 layers and 500 nodes each layer, you can specify them explicitly: `python main.py -l 5 -n 500` Similarly, if you want to change other parameters, just explicitly specify the parameters. 
* After the experiment, you can check out the complete performance of the model under the directory of `./ML_result/`. Since the model can be trained on (1) different versions of the annotations and (2) different models and model architectures, directly putting all the results within this directory will look extremely messy, so I create two folders for each of them. For example, if you use the annotations of the ISMIR paper (ISMIR2019) with the default experimental setup, the results will locate in `./ML_result/ISMIR2019/3layer300DNNwindow_size1_2training_data1timestep0ISMIR2019NCT_pitch_classpitch_class3meter_NewOnset_New_annotation_keyC__training264/`, where the folder is created with all the parameters of the experiment. Sorry for the long folder's name, but it is necessary to differentiate all the possible experimental settings since there are so many parameters to adjust. Within the folder, you can access, the training loss and validation loss for each epoch during the training process (ending with `.csv` extension), and the complete record of the model's performance (ending with `.txt` extension). Note that all the models trained on each fold of cross validation (ending with `.hdf5` extension) will be saved in the directory of `./ML_result/ISMIR2019/MODEL/`.
* If you specify the program to output the predicted results to musicXML files, you can find them under the directory of `./predicted_result/`. Similar with the last step, you can find the musicXML files in `./predicted_result/ISMIR2019/NCT_pitch_classpitch_class3meter_NewOnsetDNN1_2/`, for example. 