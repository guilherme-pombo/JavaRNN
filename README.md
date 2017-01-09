# JavaRNN
This is an implementation of a single layer generative RNN in Java (runs on CUDA)

# Dependencies

This project uses Nd4j for matrix operations (and hence can run on GPU)

# Implementation details

As of now, it is simply a single layer RNN. It uses Adagrad gradient descent, which allows an adaptable learning rate. It adapts the learning rate to the parameters, performing larger updates for infrequent and smaller updates for frequent parameters.

Adagrad's main weakness is its accumulation of the squared gradients in the denominator: Since every added term is positive, the accumulated sum keeps growing during training. This in turn causes the learning rate to shrink and eventually become infinitesimally small, at which point the algorithm is no longer able to acquire additional knowledge.

An eventual improvement to the algorithm will be to use either AdaDelta or ADAM gradient descent.

# Improvements

 - Use ADAM
 - More than single layer RNN

# Running

Just run the charRNN.java file. It will keep infinitely iterating and reducing the loss of the algorithm.
Every 100 steps, it samples from the RNN's currents state (using a seed character) and attempts to produce
text similar to the training one.
