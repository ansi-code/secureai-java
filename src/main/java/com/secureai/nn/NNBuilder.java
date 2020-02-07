package com.secureai.nn;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

public class NNBuilder {

    public FilteredMultiLayerNetwork build(int inputs, int outputs, int size) {
        int HIDDEN_SIZE = 1024;
        NeuralNetConfiguration.ListBuilder builder = new NeuralNetConfiguration.Builder()
                .seed(12345)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new Adam(0.001))
                .weightInit(WeightInit.XAVIER)
                .l1(0.001)
                .l2(0.001)
                .list()
                .layer(new DenseLayer.Builder()
                        .nIn(inputs)
                        .nOut(HIDDEN_SIZE)
                        .activation(Activation.IDENTITY)
                        .build());
        for (int i = 0; i < size; i++) {
            builder = builder.layer(new DenseLayer.Builder()
                    .nIn(HIDDEN_SIZE)
                    .nOut(HIDDEN_SIZE)
                    .activation(Activation.LEAKYRELU)
                    .build());
        }
        MultiLayerConfiguration conf = builder
                .layer(new OutputLayer.Builder(LossFunction.MSE)
                        .nIn(HIDDEN_SIZE)
                        .nOut(outputs)
                        .activation(Activation.RELU)
                        .build())
                .build();

        FilteredMultiLayerNetwork model = new FilteredMultiLayerNetwork(conf);
        model.init();
        return model;
    }
}
