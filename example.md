---
layout: default
title: Examples of the Results 

---

## Examples of the Results

The program can output the predicted results, along with the ground truth annotations, into musicXML files. The figure below is an example:

![image](https://user-images.githubusercontent.com/9313094/50618085-98b3b500-0ebe-11e9-8d8e-10ce73ea3531.png)

There are 6 rows underneath the score. The first one is the ground truth chord labels, the second one is the corresponding ground truth non-chord tones, the third one is the model’s predicted NCTs, the fourth one entails whether the predicted NCTs are correct. The fifth one is the inferred chord label (based on a heuristic algorithm I wrote), and the last one entails whether the predicted chord label is correct.
