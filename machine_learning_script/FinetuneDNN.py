import DNN_no_window
layer=[2,3,4]
nodes=[300,400,500]

for i in range(len(layer)):
    for j in range(len(nodes)):
        DNN_no_window.FineTuneDNN(layer[i],nodes[j])


#DNN_AMH_debug.FineTuneDNN(2,500)
