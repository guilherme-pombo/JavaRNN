package com.guilherme.charRNN;

import com.google.common.base.Charsets;
import com.google.common.io.Files;
import org.nd4j.linalg.api.iter.NdIndexIterator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import org.nd4j.linalg.ops.transforms.Transforms;


/**
 * This is the generative single layer RNN
 */
public class CharRNN {


    public static void main(String[] args) {
        try {
            new CharRNN();
        } catch (IOException e) {
            System.out.print(e.toString());
        }
    }

    // Hyperparameters
    private int hiddenSize;
    private int seqLength;
    private double learningRate;
    private int vocabSize;


    // RNN weights
    INDArray Wxh;
    INDArray Whh;
    INDArray Why;
    INDArray bh;
    INDArray by;

    // Adagrad memory
    INDArray mWxh;
    INDArray mWhh;
    INDArray mWhy;
    INDArray mbh;
    INDArray mby;


    private String readFile(String fileLocation) throws IOException {
        File file = new File(fileLocation);
        return Files.toString(file, Charsets.UTF_8);
    }

    private ArrayList<Character> getUniqueChars( String arg ) {
        ArrayList<Character> unique = new ArrayList<Character>();
        for( int i = 0; i < arg.length(); i++)
            if( !unique.contains( arg.charAt( i ) ) )
                unique.add( arg.charAt( i ) );
        return unique;
    }


    private CharRNN() throws IOException {
        // Read in file
        String data = readFile("input.txt");
        ArrayList<Character> chars = getUniqueChars(data);

        // Sizes
        int dataSize = data.length();
        this.vocabSize = chars.size();

        this.hiddenSize = 100;
        this.seqLength = 25;
        this.learningRate = -0.01;

        System.out.println("Size of data: " + dataSize);
        System.out.println("Size of vocabulary: " + vocabSize);

        // Maps
        HashMap<Character, Integer> CharToIx = new HashMap<Character, Integer>();
        HashMap<Integer, Character> IxToChar = new HashMap<Integer, Character>();

        for (int i = 0; i < chars.size(); i++) {
            CharToIx.put(chars.get(i), i);
            IxToChar.put(i, chars.get(i));
        }

        // Model parameters
        this.Wxh = Nd4j.rand(hiddenSize, vocabSize).mul(0.01);
        this.Whh = Nd4j.rand(hiddenSize, hiddenSize).mul(0.01);
        this.Why = Nd4j.rand(vocabSize, hiddenSize).mul(0.01);
        this.bh = Nd4j.zeros(hiddenSize, 1);
        this.by = Nd4j.zeros(vocabSize, 1);

        int n = 0;
        int p = 0;

        // Adagrad memory
        this.mWxh = Nd4j.zeros(hiddenSize, vocabSize);
        this.mWhh = Nd4j.zeros(hiddenSize, hiddenSize);
        this.mWhy = Nd4j.zeros(vocabSize, hiddenSize);
        this.mbh = Nd4j.zeros(hiddenSize, 1);
        this.mby = Nd4j.zeros(vocabSize, 1);

        // Loss at iteration 0
        double smoothLoss = - Math.log(1.0/vocabSize) * seqLength;
        INDArray hprev = Nd4j.zeros(hiddenSize, 1);

        while (true) {
            // prepare inputs (we're sweeping from left to right in steps seq_length long)
            if (p + seqLength+1 >= data.length() || n == 0) {
                // Reset the RNN memory
                hprev = Nd4j.zeros(hiddenSize, 1);
                // Start from the beggining of the data
                p = 0;
            }

            int[] inputs = new int[seqLength];
            int[] targets = new int[seqLength];

            // Create inputs and trage array
            for (int i = p; i < p + seqLength; i++) {
                inputs[i - p] = CharToIx.get(data.charAt(i));
                targets[i - p] = CharToIx.get(data.charAt(i+1));
            }


            // Sample from the model every 100 iterations
            if (n % 100 == 0) {
                int[] sampleIx = sample(hprev, inputs[0], 200);
                char[] generatedText = new char[sampleIx.length];

//                int i = 0;
//                for(int ix: sampleIx) {
//                    generatedText[i] = IxToChar.get(ix);
//                    i++;
//                }

                // System.out.println("Generated text: ");
                // System.out.println(new String(generatedText));
            }

            // forward seq_length characters through the net and fetch gradient
            Container propagation = lossFun(inputs, targets, hprev);

            // Update hidden state
            hprev = propagation.hs;

            // Update loss after forward and backprop
            smoothLoss = smoothLoss * 0.999 + propagation.loss * 0.001;

            if (n % 100 == 0) {
                System.out.println("Iteration: " + n + " Loss: " + smoothLoss);
            }

            // parameter update with Adagrad
            mWxh = mWxh.add(propagation.dWxh.muli(propagation.dWxh));
            mWhh = mWhh.add(propagation.dWhh.muli(propagation.dWhh));
            mWhy = mWhy.add(propagation.dWhy.muli(propagation.dWhy));
            mbh = mbh.add(propagation.dbh.muli(propagation.dbh));
            mby = mby.add(propagation.dby.muli(propagation.dby));

            // Adagrad update
            INDArray tmp = propagation.dWxh.mul(learningRate).div(Transforms.sqrt(mWxh.add(0.00000001)));
            Wxh = Wxh.add(tmp);

            tmp = propagation.dWhh.mul(learningRate).div(Transforms.sqrt(mWhh.add(0.00000001)));
            Whh = Whh.add(tmp);

            tmp = propagation.dWhy.mul(learningRate).div(Transforms.sqrt(mWhy.add(0.00000001)));
            Why = Why.add(tmp);

            tmp = propagation.dbh.mul(learningRate).div(Transforms.sqrt(mbh.add(0.00000001)));
            bh = bh.add(tmp);

            tmp = propagation.dby.mul(learningRate).div(Transforms.sqrt(mby.add(0.00000001)));
            by = by.add(tmp);

            // move data pointer
            p += seqLength;
            // iteration counter
            n++;
        }

    }


    private Container lossFun(int[] inputs, int[] targets, INDArray hprev) {

        HashMap<Integer, INDArray> xs =  new HashMap<Integer, INDArray>();
        HashMap<Integer, INDArray> hs =  new HashMap<Integer, INDArray>();
        HashMap<Integer, INDArray> ys =  new HashMap<Integer, INDArray>();
        HashMap<Integer, INDArray> ps =  new HashMap<Integer, INDArray>();

        // Put in the previous state
        hs.put(-1, hprev);
        double loss = 0;

        // Forward pass
        for(int t = 0; t < inputs.length; t++) {
            // Input - X
            // Generate oneHot representation
            int idx = inputs[t];
            float[] oneHotArray = new float[vocabSize];
            for (int j = 0; j < oneHotArray.length; j++) {
                if ( j == idx ) {
                    oneHotArray[j] = 1;
                }
            }
            xs.put(t, Nd4j.create(oneHotArray).transpose());
            // Hidden - H
            // Input layer to hidden
            INDArray dot1 = Wxh.mmul(xs.get(t));
            // Hidden layer to hidden
            INDArray dot2 = Whh.mmul(hs.get(t - 1)).add(bh);
            INDArray tmp = dot1.add(dot2);
            // Hidden state step, squash using ReLu
            hs.put(t, Transforms.relu(tmp));

            // Output - Y
            // Dot product between weights from h to y and hidden state, plus bias
            // unnormalized log probabilities for next chars
            ys.put(t, Why.mmul(hs.get(t)).add(by));

            // normalised Probabilities - P
            INDArray exp = Transforms.exp(ys.get(t));
            INDArray cumExp = Nd4j.sum(exp);
            ps.put(t, exp.div(cumExp));

            // Calculate the cross-entropy loss
            INDArray psState = ps.get(t);
            loss += - Math.log(psState.getDouble(targets[t], 0));
        }

        // Backward pass
        // Gradients
        INDArray dWxh = Nd4j.zeros(hiddenSize, vocabSize);
        INDArray dWhh = Nd4j.zeros(hiddenSize, hiddenSize);
        INDArray dWhy = Nd4j.zeros(vocabSize, hiddenSize);
        INDArray dbh = Nd4j.zeros(hiddenSize, 1);
        INDArray dby = Nd4j.zeros(vocabSize, 1);
        INDArray dhnext = Nd4j.zeros(hs.get(0).shape()[0], 1);

        // backward pass: compute gradients going backwards
        for (int t = inputs.length - 1; t >= 0; t--) {
            // P to y
            INDArray dy = ps.get(t);
            int idx = targets[t];
            double newValue = dy.getDouble(idx) - 1;
            dy.putScalar(new int[]{idx, 0}, newValue);

            // Y to H
            INDArray dot1 = hs.get(t).transpose().mmul(dy);
            dWhy.addi(dot1);
            dby.addi(dy);

            //Backprop into h
            dot1 = Why.transpose().mmul(dy);
            INDArray dh = dot1.add(dhnext);
            INDArray squaredH = hs.get(t).muli(hs.get(t));
            // 1 - SquareH
            NdIndexIterator iter = new NdIndexIterator(squaredH.shape());
            while (iter.hasNext()) {
                int[] nextIndex = iter.next();
                double nextVal = squaredH.getDouble(nextIndex);
                double newVal = 1 - nextVal;
                squaredH.putScalar(nextIndex, newVal);
            }

            INDArray dhraw = squaredH.muli(dh);

            dbh.add(dhraw);

            // Update gradients
            dWxh.addi(xs.get(t).transpose().mmul(dhraw));
            dWhh.addi(hs.get(t-1).transpose().mmul(dhraw));

            dhnext = Whh.transpose().mmul(dhraw);
        }

        // Clip to avoid exploding gradients
        dWxh = clipMatrix(dWxh);
        dWhy = clipMatrix(dWhy);
        dbh = clipMatrix(dbh);
        dby = clipMatrix(dby);

        return new Container(loss, dWxh, dWhh, dWhy, dbh, dby, hs.get(inputs.length - 1));
    }

    /**
     * Clip the values within a matrix to -5 to 5 range, to avoid exploding gradients
     * @param matrix
     * @return
     */
    private static INDArray clipMatrix(INDArray matrix) {
        NdIndexIterator iter = new NdIndexIterator(matrix.shape());
        while (iter.hasNext()) {
            int[] nextIndex = iter.next();
            double nextVal = matrix.getDouble(nextIndex);
            if (nextVal < -5) {
                nextVal = -5;
            }
            if (nextVal > 5) {
                nextVal = 5;
            }
            matrix.putScalar(nextIndex, nextVal);
        }
        return matrix;
    }


    /**
     * sample a sequence of integers from the model
     * h is memory state, seed_ix is seed letter for first time step
     * @param h
     * @param seedIdx
     * @param n
     * @return
     */
    private int[] sample(INDArray h, int seedIdx, int n) {
        float[] oneHotArray = new float[vocabSize];
        for (int j = 0; j < oneHotArray.length; j++) {
            if ( j == seedIdx ) {
                oneHotArray[j] = 1;
            }
        }
        INDArray x =  Nd4j.create(oneHotArray).transpose();
        int[] ixes = new int[n];

        for (int t = 0; t < n; t++) {
            INDArray dot1 = Wxh.mmul(x);
            // Hidden layer to hidden
            INDArray dot2 = Whh.mmul(h).add(bh);
            INDArray tmp = dot1.add(dot2);
            // Hidden state step, squash with rectified linear unit (can use tanh as well)
            h = Transforms.relu(tmp);

            // Output - Y
            // Dot product between weights from h to y and hidden state, plus bias
            INDArray y = Why.mmul(h).add(by);

            // Probabilities - P
            INDArray exp = Transforms.exp(y);
            INDArray cumExp = Nd4j.sum(exp);
            INDArray p = exp.div(cumExp);

            // Random choice
            int[] to_select = new int[vocabSize];
            for (int i = 0; i < vocabSize; i++){
                to_select[i] = i;
            }

            int idx = randChoice(to_select, p);

            // System.out.println("Sampled idx: " + idx);

            oneHotArray = new float[vocabSize];
            for (int j = 0; j < oneHotArray.length; j++) {
                if ( j == idx ) {
                    oneHotArray[j] = 1;
                }
            }
            x =  Nd4j.create(oneHotArray, new int[]{vocabSize, 1});
            ixes[t] = idx;
        }

        return ixes;
    }


    /**
     * Randomly select an index from a vector, given a vector of probabilities
     * @param idxs
     * @param probabilities
     * @return
     */
    private static int randChoice(int[] idxs, INDArray probabilities) {
        double p = Math.random();
        double cumulativeProbability = 0.0;
        int idx = 0;
        for (; idx < idxs.length; idx++) {
            cumulativeProbability += probabilities.getDouble(idx);
            if (p <= cumulativeProbability) {
                return idx;
            }
        }

        return idx;
    }

}
