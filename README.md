# CIFAR10
A convolutional neural net to classify the CIFAR10 set

To run the code, if you have a Theano compatible GPU:

  THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python CIFAR10.py

if not:

  THEANO_FLAGS=mode=FAST_RUN,device=cpu,floatX=float32 python CIFAR10.py

This convolutional neural net is taken from the course:
https://www.youtube.com/watch?v=S75EdAcXHKk


