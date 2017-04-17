import numpy as np
def TwoToThree(xx,x,y,z):
    k=0 # in case there is no leftover
    xxx=[[[0 for coll in range(z)] for col in range(y)] for row in range(x+1)] # for leftovers
    for i in range(0, int(xx.shape[0]/y)):
        for j in range(0, y):
            xxx[i][j] = xx[y*i+j]
    i = i+1
    if(xx.shape[0]%y!=0): # in case there are leftovers...
        for k in range(xx.shape[0]%y):
            xxx[i][k] = xx[(xx.shape[0]%y)*y+k]

        k = k+1
        for l in range(k,y): # the leftover is still list, needs to be narray
            xxx[i][l] = np.asarray(xxx[i][l])
    xxx=np.asarray(xxx)
    return xxx
