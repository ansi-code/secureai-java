package com.secureai;

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.FeedForwardLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.rng.DefaultRandom;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;
import java.util.Map;

public class DynamicExample {
    private static Logger log = LoggerFactory.getLogger(DynamicExample.class);

    public static void main(String[] args) throws Exception {
        //number of rows and columns in the input pictures
        final int numRows = 28;
        final int numColumns = 28;
        int outputNum = 10; // number of output classes
        int batchSize = 128; // batch size for each epoch
        int rngSeed = 123; // random number seed for reproducibility
        int numEpochs = 1; // number of epochs to perform
        System.setProperty("org.deeplearning4j.resources.baseurl", "https://raw.githubusercontent.com/suriyadeepan/datasets/master/toy-"); // this is a tmp fix for broken d4j blob endpoint

        //Get the DataSetIterators:
        DataSetIterator mnistTrain = new MnistDataSetIterator(batchSize, true, rngSeed);
        /*
        DataSetIterator mnistTrain = new MnistDataSetIterator(batchSize, true, rngSeed){
            @Override
            public DataSet next() {
                return super.next();
            }

            @Override
            public int totalOutcomes() {
                return 9;
            }

            @Override
            public List<String> getLabels() {
                List<String> orig = super.getLabels();
                orig.remove(orig.size()-1);
                return orig;
            }

            @Override
            public DataSet next(int num) {
                DataSet orig = super.next(num);
                orig.
                return super.next(num);
            }
        }
        */
        DataSetIterator mnistTest = new MnistDataSetIterator(batchSize, false, rngSeed);


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
                        .nOut(outputNum)
                        .activation(Activation.SOFTMAX)
                        .weightInit(WeightInit.XAVIER)
                        .build())
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        System.out.println(model.summary());

        //print the score with every 1 iteration
        model.setListeners(new ScoreIterationListener(1));

        log.info("Train model....");
        model.fit(mnistTrain, numEpochs);


        log.info("Evaluate model....");
        Evaluation eval = model.evaluate(mnistTest);
        log.info(eval.stats());
        log.info("****************Example finished********************");

        MultiLayerNetwork newModel = addOutputs(model, 1);
        System.out.println(newModel.summary());

    }

    private static MultiLayerNetwork addOutputs(MultiLayerNetwork model, int n) {
        Layer outputLayer = model.getLayer(model.getLayers().length-1);
        Map<String, INDArray> oldParamsTable = outputLayer.paramTable();
        INDArray weights = oldParamsTable.get("W");
        INDArray biases = oldParamsTable.get("b");

        oldParamsTable.put("W", Nd4j.hstack(weights, Nd4j.rand(new int[]{weights.rows(), n}).mul(-0.0001).add(0.0001)));
        oldParamsTable.put("b", Nd4j.hstack(biases, Nd4j.zeros(biases.rows(), n)));

        MultiLayerNetwork newModel = new TransferLearning.Builder(model)
                .nOutReplace(model.getLayers().length-1, weights.columns() + n, WeightInit.ONES)
                .build();

        newModel.getLayer(model.getLayers().length-1).setParamTable(oldParamsTable);

        return newModel;
    }
}
