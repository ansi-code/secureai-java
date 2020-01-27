package com.secureai;

import com.secureai.utils.ScoreWriterListener;
import org.deeplearning4j.datasets.fetchers.MnistDataFetcher;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.BaseDatasetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;

public class DynamicMNISTExample {
    private static Logger log = LoggerFactory.getLogger(DynamicMNISTExample.class);

    public static void main(String[] args) throws Exception {
        //number of rows and columns in the input pictures
        final int numRows = 28;
        final int numColumns = 28;
        int batchSize = 128; // batch size for each epoch
        int rngSeed = 123; // random number seed for reproducibility
        int numEpochs = 1; // number of epochs to perform
        System.setProperty("org.deeplearning4j.resources.baseurl", "https://raw.githubusercontent.com/suriyadeepan/datasets/master/toy-"); // this is a tmp fix for broken d4j blob endpoint

        //Get the DataSetIterators:
        DataSetIterator mnistTrain = new MnistDataSetIterator(batchSize, true, rngSeed);
        DataSetIterator mnistTest = new MnistDataSetIterator(batchSize, false, rngSeed);

        DataSetIterator mnist9Train = new Mnist9DataSetIterator(batchSize, true, rngSeed);
        DataSetIterator mnist9Test = new Mnist9DataSetIterator(batchSize, false, rngSeed);

        log.info("Build model....");
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(rngSeed) //include a random seed for reproducibility
                // use stochastic gradient descent as an optimization algorithm
                .updater(new Nesterovs(0.006, 0.9))
                .l2(1e-4)
                .list()
                .layer(new DenseLayer.Builder() //create the first, input layer with xavier initialization
                        .nIn(numRows * numColumns)
                        .nOut(1000)
                        .activation(Activation.RELU)
                        .weightInit(WeightInit.XAVIER)
                        .build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD) //create hidden layer
                        .nIn(1000)
                        .nOut(9)
                        .activation(Activation.SOFTMAX)
                        .weightInit(WeightInit.XAVIER)
                        .build())
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);

        model.init();
        System.out.println(model.summary());

        //print the score with every 1 iteration
        model.setListeners(new ScoreIterationListener(1), new ScoreWriterListener(System.getProperty("user.home") + "/d4j-data" + "/" + "1"));

        log.info("Train model....");
        model.fit(mnist9Train, numEpochs);

        log.info("Evaluate model....");
        Evaluation eval = model.evaluate(mnist9Test);
        log.info(eval.stats());
        log.info("****************Example finished********************");

        // NEW MODEL

        //MultiLayerNetwork newModel = new DynNNBuilder(model).addOutputs(1).build();
        //MultiLayerNetwork newModel = new DynNNBuilder(model).forLayer(-1).setBlockSize(1).appendOut(1).build();
        MultiLayerNetwork newModel = new TransferLearning.Builder(model).nOutReplace(model.getLayers().length - 1, 10, WeightInit.XAVIER).build();
        System.out.println(newModel.summary());

        //print the score with every 1 iteration
        newModel.setListeners(new ScoreIterationListener(1), new ScoreWriterListener(System.getProperty("user.home") + "/d4j-data" + "/" + "2"));

        log.info("Train model....");
        newModel.fit(mnistTrain, numEpochs);

        log.info("Evaluate model....");
        Evaluation eval1 = newModel.evaluate(mnistTest);
        log.info(eval1.stats());
        log.info("****************Example finished********************");
    }

    public static class Mnist9DataSetIterator extends BaseDatasetIterator {
        public Mnist9DataSetIterator(int batch, int numExamples) throws IOException {
            this(batch, numExamples, false);
        }

        public Mnist9DataSetIterator(int batch, int numExamples, boolean binarize) throws IOException {
            this(batch, numExamples, binarize, true, false, 0L);
        }

        public Mnist9DataSetIterator(int batchSize, boolean train, int seed) throws IOException {
            this(batchSize, train ? '\uea60' : 10000, false, train, true, seed);
        }

        public Mnist9DataSetIterator(int batch, int numExamples, boolean binarize, boolean train, boolean shuffle, long rngSeed) throws IOException {
            super(batch, numExamples, new MnistDataFetcher(binarize, train, shuffle, rngSeed, numExamples) {
                @Override
                public void fetch(int numExamples) {
                    super.fetch(numExamples);
                    this.curr.filterAndStrip(new int[]{0, 1, 2, 3, 4, 5, 6, 7, 8});
                }
            });
        }
    }
}
