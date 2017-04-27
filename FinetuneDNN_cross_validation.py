import DNN_no_window_cross_validation
layer=[2]
nodes=[200]

for i in range(len(layer)):
    for j in range(len(nodes)):
        DNN_no_window_cross_validation.FineTuneDNN(layer[i],nodes[j])


#DNN_AMH_debug.FineTuneDNN(2,500)
