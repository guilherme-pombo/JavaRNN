package com.guilherme.charRNN;

import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * This is simply a container for the results of forward and backprop
 */
public class Container {

    // Store the loss calculated after the forward pass
    // Gradients calculated after the backward pass
    // Hidden state
    public double loss;
    public INDArray dWxh;
    public INDArray dWhh;
    public INDArray dWhy;
    public INDArray dbh;
    public INDArray dby;
    public INDArray hs;

    public Container(double loss, INDArray dWxh, INDArray dWhh, INDArray dWhy, INDArray dbh, INDArray dby,
                     INDArray hs) {
        this.loss = loss;
        this.dWxh = dWxh;
        this.dWhh = dWhh;
        this.dWhy = dWhy;
        this.dbh = dbh;
        this.dby = dby;
        this.hs = hs;
    }

}
