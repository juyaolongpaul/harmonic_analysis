---
layout: default
title: Parameter Adjustment

---

## Parameter Adjustment

In this program, there are many ajustable parameters, ranging from the analytical styles, types of machine learning models, model architectures, and hyper-parameters. All of the parameters and the corresponding available values are shown in the chart below. The first value will be the default one. If you look at the `main.py` script, you will notice there are a few more parameters. However, they are either not essential for the task or not fully tested yet. Please let me know if you want to experiment with these parameters. 

Parameters   |Values   | Explanation
---|---|---
--source (-s)   |`rule_Maxmel`, `rule_Maxmel_NoSeventh` and many other analytical styles you can specify!   |The kind of annotations you want to use for training
--num_of_hidden_layer (-l)   |3, usually ranging from 2-5   |The number of hiddel layers (not effective in `SVM`)
--num_of_hidden_node (-n)   |300, usually ranging from 100-500  |The number of hidden nodes (not effective in `SVM`)
--model (-m)   |`DNN`, `SVM`, `LSTM` and `BLSTM` also available  |The types of models you want to use
--pitch (-p)   |`pitch_class`, `pitch_class_4_voices` also available   |The kind of pitch you want to use as features. `pitch_class` means using 12-d pitch class for each sonority; `pitch_class_4_voices` means using 12-d pitch class for each of the 4 voices
--window (-w)  |1, usually ranging from 0-5|The static window you can add as context for `DNN` or `SVM` model (not effective in `LSTM` and `BLSTM` since they can get the contextual information by specifying the `timestep`)   
--output (-o)   |`NCT`, `NCT_pitch_class` and `CL` also available|`NCT` means using 4-d output vector specifying which voice contains Non-chord-tones (NCTs), used with `pitch_class_4_voices`; `NCT_pitch_class` means using 12-d output vector specifying which pitch classes contain NCTs, used with `pitch_class`;  `CL` means training the model to predict chord directly, skipping NCT identification step. 
--input (-i)   |`3meter`, `barebone`, `2meter` and `NewOnset` also available   | Specify what input features, besides pitch, you are using. You can use meter features: `2meter` means you are using on/off beat feature; `3meter` means you are using 'strong beat, on/off beat' feature; `NewOnset` means whether the current slice has a real attack or not across all the pitch classes/voices. If used with `pitch_class`, it will add another 12-d vector in the input specifying which pitch classes are the real attacks; if used with `pitch_class_4_voices`, it will add another 4-d vector in the input specifying which voices have the real attacks
--timestep (-time)   |2, usually ranging from 2-5|`timestep` is the parameter used in `LSTM` and `BLSTM` to provide contextual information. 2 means LSTM will look at a slice before the current one as context; for BLSTM, it means the model will look a slice before and after the current one as context    
--predict (-pre)   |'Y', 'N' also available|Specify whether you want to predict and output the results in musicXML

### Usage Example

* Use DNN with 3 layers and 300 nodes each layer; 12-d pitch class, 3-d meter and the sign of real/fake attacks as input features, output as 12-d pitch class vector indicating which pitch class is NCT. Use a contextual window size of 1 and the annotation of maximally melodic and predict the results and output into musicXML file: `python main.py -l 3 -n 300 -m 'DNN' -w 1 -o 'NCT_pitch_class' -p 'pitch_class' -i '3meter_NewOnset' -pre 'Y' -time 0`. 
* Use BLSTM with the same configuration above. Only one thing to change is that the window size needs to be set as 0, and the timestep needs to be specified: `python main.py -l 3 -n 300 -m 'BLSTM' -w 0 -o 'NCT_pitch_class' -p 'pitch_class' -i '3meter_NewOnset' -pre 'Y' -time 2`
* Use DNN with the same configuration, but conduct harmonic analysis directly by skipping the identification of NCTs: `python main.py -l 3 -n 300 -m 'DNN' -w 1 -o 'CL' -p 'pitch_class' -i '3meter_NewOnset' -pre 'Y' -time 0`
* After the experiment, you can check out the complete performance of the model under the directory of `./ML_result/`. Since the model can be trained on (1) different versions of the annotations and (2) different models and model architectures, directly putting all the results within this directory will look extremely messy, so I create two folders for each of them. For example, if you use the annotations of maximally melodic with the experimental setup of `python main.py -l 3 -n 300 -m 'DNN' -w 1 -o 'NCT_pitch_class' -p 'pitch_class' -i '3meter_NewOnset' -pre 'Y' -time 0`, the results will locate in `./ML_result/rule_MaxMel/3layer300DNNwindow_size1training_data1timestep0rule_MaxMelNCT_pitch_classpitch_class3meter_NewOnset_New_annotation_keyC__training294batch_size256epochs500patience50bootstrap0balanced0/`. Sorry for the long folder's name, but it is necessary to differentiate all the possible experimental settings since there are so many parameters to adjust. Within the folder, you can access all the models trained on each fold of cross validation (ending with `.hdf5` extension), the training loss and validation loss for each epoch during the training process (ending with `.csv` extension), and the complete record of the model's performance (ending with `.txt` extension).
* If you specify the program to output the predicted results to musicXML files, you can find them under the directory of `./predicted_result/`. Similar with the last step, you can find the musicXML files in `./predicted_result/rule_MaxMel/NCT_pitch_classpitch_class3meter_NewOnsetDNN/`, for example. 