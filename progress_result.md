---
layout: default
title: Current Progress and Results 

---

## Current Progress and Results

Currently, all the experiments are conducted on 330 annotated Bach chorales in a harmonic style. The specific filters I used are:
* “Chord~sapply(Durations, min)<0.5”: Prefer chords whose durations are no less than half a beat. Without this filter, the model will occasion- ally arrange new chords on 16th beat, which is unnecessary in homorhythmic music.
* “Chord~sapply(SeventhsResolve, function(bool) any(bool %in% c(FALSE)))”: Perfer seventh chords only when the seventh note is resolved properly.
* “NCTs~Count”: Prefer chords with fewest NCTs.
* “Chord~Count”: If there are multiple analyses with the same number of NCTs, prefer the one with fewest chords. Since it prevents unnecessary chord changes, especially when a triad has a delayed sev- enth note. Instead of considering a triad changing into a seventh chord, the whole section will be la- beled with a seventh chord.
* “Chord~sapply(CompletionDelay$Durations, mean)”: If there are multiple analyses with the same number of NCTs as well as chords, prefer chords with the minimum delay of completion (e.g., all the chord tones of the chord label should appear as soon as possible).

All the experiments are using 10-fold cross validation. For evaluation metrics, I use f1-measure (abbreviated as F1) for NCT identification accuracy; frame accuracy (abbreviated as FA) to indicate the accuracy for each slice; chord accuracy (abbreviated as CA) to indicate the predicted chord accuracy compared to the ground truth annotations. CAD stands for chord accuracy using direct harmonic analysis approach; CAN stands for chord accuracy using non-chord-tone-first appraoch. The chart below specifies all the results I have got so far: The row header indicates all the experimented input features, the column header indicates all combinations between the output and different models. To save space, I use a series of acronyms for the choice of input and output architectures. Specifically:

### Acronyms for the Row Header

* I use 'PC' for pitch class. 'PC12' means 12-d pitch class category as "C, C#/Db, D, D#/Eb, E, F, F#/Gb, G, G#/Ab, A, A#/Bb, B". 'PC48' means 12-d pitch class is specified for each voice (among 4 voices). 
* I use 'M' to represent the use of 2 or 3 meter features (I did not differenciate between 2 and 3 meters since they achieve almost the same performance). 
* I use "W" to indicate the use of windows as context. Similarly, the window size of 1 or 2 usually has the similar performances, so they are not differentiated here.
* I also experiment generic pitch class as "C, D, E, F, G, A, B" and use "PC7" to represent it. "PC28" represents the generic pitch class for each voice (among 4 voices). 
* To specify whether the current slice contains a real/fake attack (onset) for a certain pitch, I use "O12" to indicate a 12-d vector specifying which pitch class contains real/fake attack by setting the value to 1 and 0, respectively; I use "O4" to incidate a 4-d vector specifying which voices contain real/fake attack. "O12" is used with "PC12/PC7" and "O4" is used with "PC48/PC28" for now. 
* I also use data augmentation in some cases. For the non-augmented approach, I tranpose all the chorales in the key of C; for the augmented appraoch, I transpose the data to 12 possible keys to increase the size of the training data, and use the data in the original key for validating and testing. I use "A" to indicate the use of data augmentation.

### Acronyms for the Column Header

* Currently, the legal chord qualities are major, minor, diminished for triads; dominant seventh, minor seventh, major seventh, half diminised seventh and fully diminished seventh for seventh chords. I also try to collapse all the seventh chords into triads in some experiments, indicated as "NO7th".
* I use "4" to indicate a 4-d vector for output that specifies which voice contains NCTs, "12" to indicate a 12-d vector for output that specifies which pitch class contains NCTs, "CL" (chord label) to indicate the appraoch of direct chord prediction skipping non-chord-tone-first approach. Consequently, the dimension of the output vector equals to the number of chord categories found in the annotations.
* For (B)LSTM models, the timestep is 2 (for best performances).
* I also ignored the number of hidden layers and hidden nodes across different models since they have little effect on the performances.
Here are the results:

### Results

Parameters   |PC12   | PC12+M|PC12+W|PC12+M+W|PC7+M+W|PC48+M+W|PC12+M+W+O12
---|---|---|---|---|---|---|---
DNN+12|f1:0.617±0.024<br/>FA:0.775±0.017|f1:0.648±0.029<br/>FA:0.787±0.019|f1:0.782±0.027<br/>FA:0.852±0.020|f1:0.815±0.025<br/>FA:0.867±0.020<br/>CA:0.852±0.021|||**f1:0.846±0.018<br/>FA:0.947±0.008<br/>CAN:0.899±0.016<br/>CAD:0.890±0.017**
DNN+12+NO7th||||**f1:0.836±0.024<br/>FA:0.882±0.018<br/>CA:0.883±0.018**|||**f1:0.840±0.023<br/>FA:0.881±0.019<br/>CA:0.884±0.019**
DNN+CL+NO7th||||**CA:0.885±0.018**|||**CA:0.887±0.018**
DNN+4||||f1:0.810±0.025<br/>FA:0.863±0.021|f1:0.799±0.020|f1:0.789±0.028<br/>FA:0.842±0.022
DNN+4<br/>Original key||||f1:0.780±0.025|
DNN+4+A||||||f1:0.794±0.024<br/>FA:0.846±0.018|
DNN+CL||||CA:0.853±0.019||CA:0.852±0.021|**CA:0.862±0.020**
SVM+CL||||CA:0.840±0.019|||
LSTM+4||||f1:0.795±0.025<br/>FA:0.856±0.019|||
BLSTM+4||||f1:0.797±0.025<br/>||f1:0.781±0.020<br/>|
BLSTM+12||||f1:0.801±0.023<br/>|||**f1:0.809±0.025<br/>FA:0.866±0.020<br/>**

### Useful Findings

* Overall, using the same input and output structures, DNN achieves the best performance, BLSTM is 0.001 consistantly lower than DNN appraoch in f1-measure; SVM has about 1.5-2% consistant lower chord accuracy compared to DNN.
* The best input combination so far is PC12+M+W+O12, reaching a f1-measure of 0.822. Notice that in this case, the chord accuracy is lower than the frame accuracy by more than 1%. To explain this, please refer to the figure below as an example: Although the NCTs are identified correctly for some slices (such as slice third and fourth in the example), it does not contain a legal chord (will be labeled as "Undetermined"), and it has to refer to the adjacent slices for the final chord label. However, if the final chord label (in the fifth slice, E chord) is different from the ground truth (E7 chord), these "Undertermined" slices (the third and fourth slices) will all inherit the wrong label, which drags down the chord accuracy performance.  

![image](https://user-images.githubusercontent.com/9313094/50726953-0db30480-10e2-11e9-9d8e-be22368a25cc.png)

* Results show that if only PC12 is used as input feature on DNN+12, f1-measure is only 0.617, but with a small window as context, the performance boosts significantly to 0.782, and with the meter features, it further improves to 0.815. By specifying the sign of real/fake attack on 12 pitch class, the performance further improves to 0.822.
* Results show that using pitch class for 4 voices (to incorporate more voice leading infomation) actually undermines the performance by about 0.002 in f1-measure, since it causes the problem of overfitting. Therefore, we need more training data in order to use these features. 
* By collapsing 7th chord into triads, the performance further improves into 0.836 in f1-measure, and frame accuracy and chord accuracy is above 88%.
* Overall, the result are promising, and even when you look at the errors the model makes -- although some of them might not be perfectly idiosyncratic in the style of maximally melodic, most of them are still acceptable chord labels. 
