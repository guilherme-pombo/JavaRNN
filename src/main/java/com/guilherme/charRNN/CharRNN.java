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
    private int hiddenSize = 100;
    private int seqLength = 25;
    private double learningRate = 0.1;


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
        int vocabSize = chars.size();

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
        INDArray Wxh = Nd4j.rand(hiddenSize, vocabSize);
        INDArray Whh = Nd4j.rand(hiddenSize, hiddenSize);
        INDArray Why = Nd4j.rand(vocabSize, hiddenSize);
        INDArray bh = Nd4j.zeros(hiddenSize, 1);
        INDArray by = Nd4j.zeros(vocabSize, 1);

        int n = 0;
        int p = 0;

        // Adagrad memory
        INDArray mWxh = Nd4j.zeros(hiddenSize, vocabSize);
        INDArray mWhh = Nd4j.zeros(hiddenSize, hiddenSize);
        INDArray mWhy = Nd4j.zeros(vocabSize, hiddenSize);
        INDArray mbh = Nd4j.zeros(hiddenSize, 1);
        INDArray mby = Nd4j.zeros(vocabSize, 1);

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
                int[] sampleIx = sample(hprev, inputs[0], n, Wxh, Whh, Why, bh, by, vocabSize);
                char[] generatedText = new char[sampleIx.length];

                int i = 0;
                for(int ix: sampleIx) {
                    generatedText[i] = IxToChar.get(ix);
                    i++;
                }

                System.out.println("Generated text: ");
                System.out.println(new String(generatedText));
            }

            // forward seq_length characters through the net and fetch gradient
            Container propagation = lossFun(inputs, targets, hprev, Wxh, Whh, Why, bh, by, vocabSize, hiddenSize);

            // Update loss after forward and backprop
            smoothLoss = smoothLoss * 0.999 + propagation.loss * 0.001;

            if (n % 100 == 0) {
                System.out.println("Iteration: " + n + " Loss: " + smoothLoss);
            }

            // parameter update with Adagrad
            mWxh = mWxh.add(Wxh.muli(Wxh));
            mWhh = mWhh.add(Whh.muli(Whh));
            mWhy = mWhy.add(Why.muli(Why));
            mbh = mbh.add(bh.muli(bh));
            mby = mby.add(by.muli(by));

            // Adagrad update
            Wxh = propagation.dWxh.mul(- learningRate).div(Transforms.sqrt(mWxh.add(0.00000001)));
            Whh = propagation.dWhh.mul(- learningRate).div(Transforms.sqrt(mWhh.add(0.00000001)));
            Why = propagation.dWhy.mul(- learningRate).div(Transforms.sqrt(mWhy.add(0.00000001)));
            bh = propagation.dbh.mul(- learningRate).div(Transforms.sqrt(mbh.add(0.00000001)));
            by = propagation.dby.mul(- learningRate).div(Transforms.sqrt(mby.add(0.00000001)));

            // move data pointer
            p += seqLength;
            // iteration counter
            n++;
        }

    }


    private static Container lossFun(int[] inputs, int[] targets, INDArray hprev, INDArray Wxh, INDArray Whh,
                               INDArray Why, INDArray bh, INDArray by, int vocabSize, int hiddenSize) {

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
            // Hidden state step, squash between -1 and 1 with hyperbolic tan
            hs.put(t, Transforms.tanh(tmp));

            // Output - Y
            // Dot product between weights from h to y and hidden state, plus bias
            ys.put(t, Why.mmul(hs.get(t)).add(by));

            // Probabilities - P
            INDArray exp = Transforms.exp(ys.get(t));
            INDArray cumExp = exp.cumsum(0);
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
            INDArray dot1 = dy.mmul(hs.get(t).transpose());
            dWhy.add(dot1);
            dby.add(dy);

            //Backprop into h
            dot1 = Why.transpose().mmul(dy);
            INDArray dh = dot1.add(dhnext);
            INDArray squaredH = hs.get(t).muli(hs.get(t));
            INDArray ones = Nd4j.ones(squaredH.shape());
            INDArray dhraw = dh.muli(ones.sub(squaredH));

            dbh.add(dhraw);

            // Update gradients
            dWxh.add(dhraw.mmul(xs.get(t).transpose()));
            dWhh.add(dhraw.mmul(hs.get(t-1).transpose()));

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
    private static int[] sample(INDArray h, int seedIdx, int n, INDArray Wxh, INDArray Whh,
                               INDArray Why, INDArray bh, INDArray by, int vocabSize) {
        float[] oneHotArray = new float[vocabSize];
        for (int j = 0; j < oneHotArray.length; j++) {
            if ( j == seedIdx ) {
                oneHotArray[j] = 1;
            }
        }
        INDArray x =  Nd4j.create(oneHotArray).transpose();
        System.out.println(x.toString());
        int[] ixes = new int[n];

        for (int t = 0; t < n; t++) {
            INDArray dot1 = Wxh.mmul(x);
            // Hidden layer to hidden
            INDArray dot2 = Whh.mmul(h).add(bh);
            INDArray tmp = dot1.add(dot2);
            // Hidden state step, squash between -1 and 1 with hyperbolic tan
            h = Transforms.tanh(tmp);

            // Output - Y
            // Dot product between weights from h to y and hidden state, plus bias
            INDArray y = Why.mmul(h).add(by);

            // Probabilities - P
            INDArray exp = Transforms.exp(y);
            INDArray cumExp = exp.cumsum(0);
            INDArray p = exp.div(cumExp);

            // Random choice
            int[] to_select = new int[vocabSize];
            for (int i = 0; i < vocabSize; i++){
                to_select[i] = i;
            }

            int idx = randChoice(to_select, p);

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
