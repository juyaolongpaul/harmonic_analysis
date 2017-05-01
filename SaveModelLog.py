import time
import h5py
def Save(name, hist, model, test_xx, test_yy, windowsize, portion):
    log=open(name + '.txt','w+')
    log.write(time.ctime())
    log.write('\n')
    '''for item, value in hist.params.iteritems():
        log.write(item)
        log.write(' ')
        log.write(str(value))
        log.write('\n')
        #print (item, value, file=log)
    for item, value in hist.history.iteritems():
        log.write(item)
        log.write(' ')
        log.write(str(value))
        log.write('\n')
        #print (item, value, file=log)'''

    json_string = model.to_json()
    open(name+'.json','w').write(json_string)
    model.save_weights(name+'.h5')
    score = model.evaluate(test_xx, test_yy, verbose=0)
    print('Test loss:', score[0], file= log)
    print('Test accuracy:', score[1], file= log)
    log.close()