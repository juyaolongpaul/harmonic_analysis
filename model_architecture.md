---
layout: default
title: Model's Architecture (in the Example of DNN) 

---

## The Model's Architecture (in the Example of DNN)

For input and output encoding, I use the one-hot encoding method. The example uses 12-d pitch class, 2 meter features to indicate the current slice (highlighted in the solid line) being on/off beat and the window size 1 to add the previous slice and the following slice (highlighted in the dashed line) as context, along with 12-d pitch class for output indicating which pitch classes are NCTs, the resulting model's architecture looks like this:
![image](https://user-images.githubusercontent.com/9313094/50612164-081daa80-0ea7-11e9-85d1-b46246f7ae5f.png)
Other experimental settings are shown here:
![image](https://user-images.githubusercontent.com/9313094/50612318-91cd7800-0ea7-11e9-84ba-f1dc5fd6bb9b.png)