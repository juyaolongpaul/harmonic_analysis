import DNN_no_window
layer=[1,2,3]
nodes=[100]

for i in range(len(layer)):
    for j in range(len(nodes)):
        DNN_no_window.FineTuneDNN(layer[i], nodes[j], 1, 0.1)
        DNN_no_window.FineTuneDNN(layer[i], nodes[j], 1, 0.2)
        DNN_no_window.FineTuneDNN(layer[i], nodes[j], 1, 0.3)
        DNN_no_window.FineTuneDNN(layer[i], nodes[j], 1, 0.4)
        DNN_no_window.FineTuneDNN(layer[i], nodes[j], 1, 0.5)
        DNN_no_window.FineTuneDNN(layer[i], nodes[j], 1, 0.6)
        DNN_no_window.FineTuneDNN(layer[i], nodes[j], 1, 0.7)
        DNN_no_window.FineTuneDNN(layer[i], nodes[j], 1, 0.8)
        DNN_no_window.FineTuneDNN(layer[i], nodes[j], 1, 0.9)
        DNN_no_window.FineTuneDNN(layer[i],nodes[j],1,1)


#DNN_AMH_debug.FineTuneDNN(2,500)
