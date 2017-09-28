import DNN_no_window
layer=[2]
nodes=[200]

for i in range(len(layer)):
    for j in range(len(nodes)):
        #DNN_no_window.FineTuneDNN(layer[i], nodes[j], 2, 0.1)
        #DNN_no_window.FineTuneDNN(layer[i], nodes[j], 2, 0.2)
        #DNN_no_window.FineTuneDNN(layer[i], nodes[j], 2, 0.3)
        #DNN_no_window.FineTuneDNN(layer[i], nodes[j], 2, 0.4)
        #DNN_no_window.FineTuneDNN(layer[i], nodes[j], 2, 0.5)
        #DNN_no_window.FineTuneDNN(layer[i], nodes[j], 2, 0.6)
        #DNN_no_window.FineTuneDNN(layer[i], nodes[j], 2, 0.7)
        #DNN_no_window.FineTuneDNN(layer[i], nodes[j], 2, 0.8)
        #DNN_no_window.FineTuneDNN(layer[i], nodes[j], 2, 0.9)
        #DNN_no_window.FineTuneDNN_non_chord_tone_shuffle(layer[i], nodes[j], 2, 1, 10)
        #DNN_no_window.FineTuneDNN_non_chord_tone_shuffle(layer[i], nodes[j], 3, 1, 10)
        #DNN_no_window.FineTuneDNN_non_chord_tone_shuffle(layer[i], nodes[j], 4, 1, 10)
        #DNN_no_window.FineTuneDNN_non_chord_tone_shuffle(layer[i], nodes[j], 5, 1, 10)
        DNN_no_window.FineTuneDNN_non_chord_tone(layer[i], nodes[j], 0, 1)
        #DNN_no_window.FineTuneDNN_non_chord_tone(layer[i], nodes[j], 1, 1)
        #DNN_no_window.FineTuneDNN_non_chord_tone(layer[i], nodes[j], 1, 0.2)
        #DNN_no_window.FineTuneDNN_non_chord_tone(layer[i], nodes[j], 1, 0.3)
        #DNN_no_window.FineTuneDNN_non_chord_tone(layer[i], nodes[j], 1, 0.4)
        #DNN_no_window.FineTuneDNN_non_chord_tone(layer[i], nodes[j], 1, 0.5)
        #DNN_no_window.FineTuneDNN_non_chord_tone(layer[i], nodes[j], 1, 0.6)
        #DNN_no_window.FineTuneDNN_non_chord_tone(layer[i], nodes[j], 1, 0.7)
        #DNN_no_window.FineTuneDNN_non_chord_tone(layer[i], nodes[j], 1, 0.8)
        DNN_no_window.FineTuneDNN_non_chord_tone(layer[i], nodes[j], 1, 1)
        DNN_no_window.FineTuneDNN_non_chord_tone(layer[i], nodes[j], 2, 1)
        DNN_no_window.FineTuneDNN_non_chord_tone(layer[i], nodes[j], 3, 1)
        DNN_no_window.FineTuneDNN_non_chord_tone(layer[i], nodes[j], 4, 1)
        DNN_no_window.FineTuneDNN_non_chord_tone(layer[i], nodes[j], 5, 1)
        #DNN_no_window.FineTuneDNN_non_chord_tone(layer[i], nodes[j], 2, 1)
        #DNN_no_window.FineTuneDNN_non_chord_tone(layer[i], nodes[j], 3, 1)
        #DNN_no_window.FineTuneDNN(layer[i],nodes[j],1,1)
        #DNN_no_window.FineTuneDNN(layer[i], nodes[j], 2, 1)
        #DNN_no_window.FineTuneDNN(layer[i],nodes[j],3,1)
        #DNN_no_window.FineTuneDNN(layer[i],nodes[j],4,1)
        #DNN_no_window.FineTuneDNN(layer[i],nodes[j],5,1)


#DNN_AMH_debug.FineTuneDNN(2,500)
