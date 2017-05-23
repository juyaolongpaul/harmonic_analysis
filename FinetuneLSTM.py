import LSTM_AMH_TRAIN_BATCH
layer=[2]
nodes=[200]
batch_size = [20]
for i in range(len(layer)):
    for j in range(len(nodes)):
        for k in range(len(batch_size)):
            LSTM_AMH_TRAIN_BATCH.TrainBatch(layer[i],nodes[j],batch_size[k], 1, 1)

