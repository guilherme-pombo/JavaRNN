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

    // -- Hyperparameters of RNN --
    // Number of nodes in the hidden layer
    private int hiddenSize;
    // Number of characters to take in at a time
    private int seqLength;
    // Learning rate, not really important since we are using Adagrad which has a dynamic learning rate
    // Will just set to 0.01
    private double learningRate;
    // Number of unique characters in the training data
    private int vocabSize;


    // RNN weights
    // Weights from input layer to hidden layer
    private INDArray Wxh;
    // Weights from hidden layer nodes to hidden layer nodes
    private INDArray Whh;
    // Weights from hidden layer to output layer
    private INDArray Why;
    // Hidden layer bias
    private INDArray bh;
    // Output layer bias
    private INDArray by;


    /**
     * Read a file into a string
     * @param fileLocation
     * @return
     * @throws IOException
     */
    private String readFile(String fileLocation) throws IOException {
        File file = new File(fileLocation);
        return Files.toString(file, Charsets.UTF_8);
    }

    /**
     * Get an ArrayList with the unique characters that occur in a string
     * @param arg
     * @return
     */
    private ArrayList<Character> getUniqueChars( String arg ) {
        ArrayList<Character> unique = new ArrayList<Character>();
        for( int i = 0; i < arg.length(); i++)
            if( !unique.contains( arg.charAt( i ) ) )
                unique.add( arg.charAt( i ) );
        return unique;
    }


    /**
     * This is the constructor for the RNN
     * It keeps on training the RNN infinitely, sampling random text from the RNN model every 100 iterations
     * The loss should keeps going down, until supposedly reaching 0
     * This is obviously not the best way to train a RNN
     * Only for illustration purposes that the text sampled from the RNN gets more and more congruent and similar
     * to the original training data as the epochs go on
     * @throws IOException
     */
    private CharRNN() throws IOException {
        // Read in file
        String data = readFile("input.txt");
        // Get unique chars in the text file
        ArrayList<Character> chars = getUniqueChars(data);

        // Number of characters in the text
        int dataSize = data.length();
        // Number of unique characters in the text
        this.vocabSize = chars.size();

        // Define the RNN hyperparameters
        this.hiddenSize = 100;
        this.seqLength = 25;
        this.learningRate = 0.01;

        System.out.println("Size of data: " + dataSize);
        System.out.println("Size of vocabulary: " + vocabSize);

        // Character to integer map
        HashMap<Character, Integer> CharToIx = new HashMap<Character, Integer>();
        // Integer to character map
        HashMap<Integer, Character> IxToChar = new HashMap<Integer, Character>();

        for (int i = 0; i < chars.size(); i++) {
            CharToIx.put(chars.get(i), i);
            IxToChar.put(i, chars.get(i));
        }

        // Create the weights of the RNN to have small random values
        this.Wxh = Nd4j.randn(hiddenSize, vocabSize).mul(0.01);
        this.Whh = Nd4j.randn(hiddenSize, hiddenSize).mul(0.01);
        this.Why = Nd4j.randn(vocabSize, hiddenSize).mul(0.01);
        this.bh = Nd4j.zeros(hiddenSize, 1);
        this.by = Nd4j.zeros(vocabSize, 1);

        // This stores the number of iterations performed
        int n = 0;
        // This a pointer to where in the text we are
        int p = 0;

        // Adagrad memory -- they each are
        // a diagonal matrix where each diagonal element is the sum of the squares of the gradients
        // They serve as the adaptive learning rate
        INDArray mWxh = Nd4j.zeros(hiddenSize, vocabSize);
        INDArray mWhh = Nd4j.zeros(hiddenSize, hiddenSize);
        INDArray mWhy = Nd4j.zeros(vocabSize, hiddenSize);
        INDArray mbh = Nd4j.zeros(hiddenSize, 1);
        INDArray mby = Nd4j.zeros(vocabSize, 1);

        // Loss at iteration 0
        double smoothLoss = - Math.log(1.0/vocabSize) * seqLength;
        // Initial value of RNN memory
        INDArray hprev = Nd4j.zeros(hiddenSize, 1);

        // Keep iterating forever and sampling next text every 100 iterations
        while (true) {
            // prepare inputs (we're sweeping from left to right in steps seq_length long)
            if (p + seqLength+1 >= data.length() || n == 0) {
                // Reset the RNN memory
                hprev = Nd4j.zeros(hiddenSize, 1);
                // Start from the first character in the training text
                p = 0;
            }

            // This will store the index of the input characters
            int[] inputs = new int[seqLength];
            // This will store the next character in a sequence
            // If the text is "Hello" and input is "H", the target is "e"
            int[] targets = new int[seqLength];

            // Create inputs and target array
            for (int i = p; i < p + seqLength; i++) {
                inputs[i - p] = CharToIx.get(data.charAt(i));
                targets[i - p] = CharToIx.get(data.charAt(i+1));
            }


            // Sample new text from the model every 100 iterations
            if (n % 100 == 0) {
                // Get the ids of the characters
                int[] sampleIx = sample(hprev, inputs[0], 100);
                char[] generatedText = new char[sampleIx.length];

                int i = 0;
                for(int ix: sampleIx) {
                    generatedText[i] = IxToChar.get(ix);
                    i++;
                }

                System.out.println("Generated text: ");
                System.out.println(new String(generatedText));
            }

            // Execute forward and backwards pass on the RNN and get
            // the loss at this stage, as well as the gradients for all the parameters
            // and the new hidden state
            Container res = lossFun(inputs, targets, hprev);

            // Update hidden state
            hprev = res.hs;

            // Update loss after forward and backprop
            smoothLoss = smoothLoss * 0.999 + res.loss * 0.001;

            if (n % 100 == 0) {
                System.out.println("Iteration: " + n + " Loss: " + smoothLoss);
            }

            // parameter update with Adagrad
            // Again, the Adagrad memory stores the square of the gradient for each parameter
            mWxh.addi(res.dWxh.mul(res.dWxh));
            mWhh.addi(res.dWhh.mul(res.dWhh));
            mWhy.addi(res.dWhy.mul(res.dWhy));
            mbh.addi(res.dbh.mul(res.dbh));
            mby.addi(res.dby.mul(res.dby));


            // Adagrad update -
            // add smoothing parameter, 0.0000001, to the squared gradients to avoid division by zero
            // Weights = old_weights - learningRate*gradients/sqrt(squared_gradients + 1e-7)

            Wxh.subi((res.dWxh.mul(learningRate))
                    .div(Transforms.sqrt(mWxh.add(0.0000001))));

            Whh.subi((res.dWhh.mul(learningRate))
                    .div(Transforms.sqrt(mWhh.add(0.0000001))));

            Why.subi((res.dWhy.mul(learningRate))
                    .div(Transforms.sqrt(mWhy.add(0.0000001))));

            bh.subi((res.dbh.mul(learningRate))
                    .div(Transforms.sqrt(mbh.add(0.0000001))));

            by.subi((res.dby.mul(learningRate))
                    .div(Transforms.sqrt(mby.add(0.0000001))));

            // move data pointer
            p += seqLength;
            // iteration counter
            n++;
        }

    }


    /**
     * Given inputs and targets, this performs the forward and backward pass for the RNN
     * Returns the loss after the forward pass
     * The gradients for each parameter of the model, calculated post backward pass
     * The current hidden state
     * @param inputs
     * @param targets
     * @param hprev
     * @return
     */
    private Container lossFun(int[] inputs, int[] targets, INDArray hprev) {

        // Stores inputs
        HashMap<Integer, INDArray> xs =  new HashMap<Integer, INDArray>();
        // Stores hidden states
        HashMap<Integer, INDArray> hs =  new HashMap<Integer, INDArray>();
        // Stores outputs of RNN
        HashMap<Integer, INDArray> ys =  new HashMap<Integer, INDArray>();
        // Stores normalised outputs -- probabilities of each character
        HashMap<Integer, INDArray> ps =  new HashMap<Integer, INDArray>();

        // Put in the previous state
        hs.put(-1, hprev);
        double loss = 0;

        // Forward pass
        for(int t = 0; t < inputs.length; t++) {
            // Store one-hot encoding of the inputs
            int idx = inputs[t];
            xs.put(t, oneHotEncoding(idx));

            // Hidden layer state
            // Input layer to hidden
            INDArray dot1 = Wxh.mmul(xs.get(t));
            // Hidden layer to hidden -- add bias
            INDArray dot2 = Whh.mmul(hs.get(t - 1)).add(bh);
            // Hidden state step, squash using tanh to -1 to 1
            hs.put(t, Transforms.tanh(dot1.add(dot2)));

            // Outputs
            // Hidden state (dot product) with weights from hidden to output (add bias as well)
            ys.put(t, Why.mmul(hs.get(t)).add(by));

            // normalised Probabilities - P
            INDArray exp = Transforms.exp(ys.get(t));
            // Normalisation factor
            INDArray cumExp = Nd4j.sum(exp);
            ps.put(t, exp.div(cumExp));

            // Given the target and the probabilities of the each character
            // Calculate the cross-entropy loss
            loss += - Math.log(ps.get(t).getDouble(targets[t], 0));
        }

        // Backward pass
        // Gradients for each parameter
        INDArray dWxh = Nd4j.zeros(hiddenSize, vocabSize);
        INDArray dWhh = Nd4j.zeros(hiddenSize, hiddenSize);
        INDArray dWhy = Nd4j.zeros(vocabSize, hiddenSize);
        INDArray dbh = Nd4j.zeros(hiddenSize, 1);
        INDArray dby = Nd4j.zeros(vocabSize, 1);
        INDArray dhnext = Nd4j.zeros(hs.get(0).shape()[0], 1);

        // backward pass: compute gradients starting with the probabilities and going back
        // to output layer, then hidden, then input layer
        for (int t = inputs.length - 1; t >= 0; t--) {

            // Probabilities to output layer
            INDArray dy = ps.get(t);
            int idx = targets[t];
            double newValue = dy.getDouble(idx) - 1;
            dy.putScalar(new int[]{idx, 0}, newValue);

            INDArray dot1 = dy.mmul(hs.get(t).transpose());
            // Gradient for hidden to output
            dWhy.addi(dot1);
            // Output Bias gradient
            dby.addi(dy);

            // Backprop into the hidden layer
            dot1 = Why.transpose().mmul(dy);
            // Gradient for hidden layer bias
            INDArray dh = dot1.add(dhnext);

            INDArray squaredH = Transforms.pow(hs.get(t),2);
            // 1 - SquareH
            NdIndexIterator iter = new NdIndexIterator(squaredH.shape());
            while (iter.hasNext()) {
                int[] nextIndex = iter.next();
                double nextVal = squaredH.getDouble(nextIndex);
                double newVal = 1 - nextVal;
                squaredH.putScalar(nextIndex, newVal);
            }

            INDArray dhraw = squaredH.mul(dh);

            dbh.addi(dhraw);

            // Hidden to hidden weight gradients
            dWhh.addi(dhraw.mmul(hs.get(t-1).transpose()));

            // Output to hidden weight gradients
            dWxh.addi(dhraw.mmul(xs.get(t).transpose()));

            dhnext = Whh.transpose().mmul(dhraw);
        }

        // Clip to avoid exploding gradients
        // Don't have to worry about underflowing gradients with RNN and tanh activation
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
     * Generate a one hot encoding for an input
     * A one hot encoding is a matrix where all elements are 0, except for one entry which is 1
     * So say we want to to encode letter "c" which is the 4th element in our vocabulary of 5 elements we write
     * [0,0,0,1,0]
     * @param idx
     * @return
     */
    private INDArray oneHotEncoding(int idx) {
        float[] oneHotArray = new float[vocabSize];
        for (int j = 0; j < oneHotArray.length; j++) {
            if ( j == idx ) {
                oneHotArray[j] = 1;
            }
        }
        return Nd4j.create(oneHotArray).transpose();
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

        // Use a seed character to start the sampling from
        INDArray x =  oneHotEncoding(seedIdx);
        int[] ixes = new int[n];

        // Do forward pass
        for (int t = 0; t < n; t++) {
            // Input to hidden
            INDArray dot1 = Wxh.mmul(x);
            // Hidden layer to hidden
            INDArray dot2 = Whh.mmul(h).add(bh);
            // Hidden state step, squash with tanh to -1 to 1
            h = Transforms.tanh(dot1.add(dot2));

            // Output - Y
            // Dot product between weights from h to y and hidden state, plus bias
            INDArray y = Why.mmul(h).add(by);

            // Normalised Probabilities - P
            INDArray exp = Transforms.exp(y);
            INDArray cumExp = Nd4j.sum(exp);
            INDArray p = exp.div(cumExp);

            int[] to_select = new int[vocabSize];
            for (int i = 0; i < vocabSize; i++){
                to_select[i] = i;
            }

            // Given the probabilities of the characters, pick "random characters" to generate the text
            int idx = randChoice(to_select, p);

            // Next character in the sequence
            x = oneHotEncoding(idx);
            // Store the chosen character
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
