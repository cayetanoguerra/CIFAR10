import cPickle

def saveModel(params,fileName='best_model.pkl'):
    f = open(fileName,'wb')
    cPickle.dump(params, f)
    f.close()

def loadModel(fileName='best_model.pkl'):
    f = open(fileName,'rb')
    params = cPickle.load(f)
    f.close()
    return params