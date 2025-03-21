# NNFS
This is my implementation of a convolutional neural network from scratch to classify the CIFAR-10 dataset. To run it, install the requirements with `pip install -r requirements.txt` and the run the file with `python convnet.py.`. It contains all of the code as well as a test demonstration of my best model. 

The design is inspired by pytorch, with each component getting its own class, each with a `forward`, `backward`, and `__call__ `fuction. A model is declared by defining its comonents in its `__init__` as well as the order of the layers in an array `layers`. The funciton `forward` defines the forward pass of the model given an input.
