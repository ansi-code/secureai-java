package com.secureai.nn;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class NNBuilder {

    public FilteredMultiLayerNetwork build(int inputs, int outputs, int size) {
        int HIDDEN_SIZE = 128;
        NeuralNetConfiguration.ListBuilder builder = new NeuralNetConfiguration.Builder()
                .updater(new Adam(1e-3))
                .l1(1e-4)
                .l2(1e-4)
                .list()
                .layer(new DenseLayer.Builder()
                        .nIn(inputs)
                        .nOut(HIDDEN_SIZE)
                        .activation(Activation.IDENTITY)
                        .weightInit(WeightInit.XAVIER)
                        .build());
        for (int i = 0; i < size; i++) {
            builder = builder.layer(new DenseLayer.Builder()
                    .nIn(HIDDEN_SIZE)
                    .nOut(HIDDEN_SIZE)
                    .activation(Activation.RELU)
                    .weightInit(WeightInit.XAVIER)
                    .build());
        }
        MultiLayerConfiguration conf = builder
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nIn(HIDDEN_SIZE)
                        .nOut(outputs)
                        .activation(Activation.SOFTMAX)
                        .weightInit(WeightInit.XAVIER)
                        .build())
                .build();

        FilteredMultiLayerNetwork model = new FilteredMultiLayerNetwork(conf);
        model.init();
        return model;
    }
}
