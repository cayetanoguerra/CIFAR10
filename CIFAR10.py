import numpy as np
import theano

from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from theano.tensor.nnet.conv import conv2d
from theano.tensor.signal.downsample import max_pool_2d

import gzip
import cPickle

srng = RandomStreams()

# Deserialize the data file
def unpickle(file):
    import cPickle
    fo = open(file, 'rb')
    dictionary = cPickle.load(fo)
    fo.close()
    return dictionary

# Translate a list of labels into an array of 0's and one 1.
# i.e.: 4 -> [0,0,0,0,1,0,0,0,0,0] 
def one_hot(x, n):
    if type(x) == list:
        x = np.array(x)
    x = x.flatten()
    o_h = np.zeros((len(x), n))
    o_h[np.arange(len(x)), x] = 1
    return o_h
    
# Translate a list into a Numpy array
def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)

# Initialize the weights of every layer. The "*" means converting a 
# list in a sequence of parameters
def init_weights(shape):
    return theano.shared(floatX(np.random.randn(*shape) * 0.01))

# Activation function
def rectify(X):
    return T.maximum(X, 0.)

def softmax(X):
    e_x = T.exp(X - X.max(axis=1).dimshuffle(0, 'x'))
    return e_x / e_x.sum(axis=1).dimshuffle(0, 'x')

# Dropout is a way to reduce overfitting
def dropout(X, p=0.):
    if p > 0:
        retain_prob = 1 - p
        X *= srng.binomial(X.shape, p=retain_prob, dtype=theano.config.floatX)
        X /= retain_prob
    return X

# Parameters updating function
def RMSprop(cost, params, lr=0.001, rho=0.9, epsilon=1e-6):
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    for p, g in zip(params, grads):
        acc = theano.shared(p.get_value() * 0.)
        acc_new = rho * acc + (1 - rho) * g ** 2
        gradient_scaling = T.sqrt(acc_new + epsilon)
        g = g / gradient_scaling
        updates.append((acc, acc_new))
        updates.append((p, p - lr * g))
    return updates

def model(X, w, w2, w3, w4, p_drop_conv, p_drop_hidden):
    l1a = rectify(conv2d(X, w, border_mode='full'))
    l1 = max_pool_2d(l1a, (2, 2))
    l1 = dropout(l1, p_drop_conv)

    l2a = rectify(conv2d(l1, w2))
    l2 = max_pool_2d(l2a, (2, 2))
    l2 = dropout(l2, p_drop_conv)

    l3a = rectify(conv2d(l2, w3))
    l3b = max_pool_2d(l3a, (2, 2))
    l3 = T.flatten(l3b, outdim=2)
    l3 = dropout(l3, p_drop_conv)

    l4 = rectify(T.dot(l3, w4))
    l4 = dropout(l4, p_drop_hidden)

    pyx = softmax(T.dot(l4, w_o))
    return l1, l2, l3, l4, pyx


print "\n Loading the CIFAR-10 set..."

labels = unpickle("cifar-10-batches-py/batches.meta")

cifar_train_1 = unpickle("cifar-10-batches-py/data_batch_1")
cifar_train_2 = unpickle("cifar-10-batches-py/data_batch_2")
cifar_train_3 = unpickle("cifar-10-batches-py/data_batch_3")
cifar_train_4 = unpickle("cifar-10-batches-py/data_batch_4")
cifar_train_5 = unpickle("cifar-10-batches-py/data_batch_5")

cifar_test = unpickle("cifar-10-batches-py/test_batch")

trX = np.concatenate((cifar_train_1["data"], cifar_train_2["data"], cifar_train_3["data"], cifar_train_4["data"], cifar_train_5["data"]))

trY = np.asarray(cifar_train_1["labels"] + cifar_train_2["labels"] + cifar_train_3["labels"] + cifar_train_4["labels"] + cifar_train_5["labels"])

teX = cifar_test["data"]
teY = np.asarray(cifar_test["labels"])


print "-------"
print trX.shape
print trY.shape
print teX.shape
print teY.shape


trY = one_hot(trY, 10)
teY = one_hot(teY, 10)

trX = trX.reshape(-1, 3, 32, 32)
teX = teX.reshape(-1, 3, 32, 32)

X = T.ftensor4()
Y = T.fmatrix()

w = init_weights((32, 3, 3, 3))
w2 = init_weights((64, 32, 3, 3))
w3 = init_weights((128, 64, 3, 3))
w4 = init_weights((128 * 3 * 3, 625))
w_o = init_weights((625, 10))

noise_l1, noise_l2, noise_l3, noise_l4, noise_py_x = model(X, w, w2, w3, w4, 0.2, 0.5)
l1, l2, l3, l4, py_x = model(X, w, w2, w3, w4, 0., 0.)
y_x = T.argmax(py_x, axis=1)


cost = T.mean(T.nnet.categorical_crossentropy(noise_py_x, Y))
params = [w, w2, w3, w4, w_o]
updates = RMSprop(cost, params, lr=0.001)

train = theano.function(inputs=[X, Y], outputs=cost, updates=updates, allow_input_downcast=True)
predict = theano.function(inputs=[X], outputs=y_x, allow_input_downcast=True)

f = open('log01_20samples_CIFAR_01.txt', 'w')

print "---------------------------------------"

for i in range(100):
    for start, end in zip(range(0, len(trX), 128), range(128, len(trX), 128)):
        cost = train(trX[start:end], trY[start:end])
    result = np.mean(np.argmax(teY, axis=1) == predict(teX))
    print result
    f.write("%s\n" % str(result))
    f.flush()
    
f.close()
    
