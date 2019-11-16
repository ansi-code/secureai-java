package com.secureai.nn;

import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;
import java.util.Map;

public class DynNeuralNetworkBuilder {

    private MultiLayerNetwork model;

    public DynNeuralNetworkBuilder(MultiLayerNetwork model) {
        this.model = model;
    }

    private DynNeuralNetworkBuilder addOutputs(int n) {
        Layer outputLayer = this.model.getLayer(this.model.getLayers().length - 1);

        Map<String, INDArray> oldParamsTable = outputLayer.paramTable();
        INDArray weights = oldParamsTable.get("W");
        INDArray biases = oldParamsTable.get("b");
        //System.out.println(Arrays.toString(outputLayer.params().shape()));
        //System.out.println(Arrays.toString(weights.shape()));
        //System.out.println(Arrays.toString(biases.shape()));

        oldParamsTable.put("W", Nd4j.hstack(weights, Nd4j.rand(new int[]{weights.rows(), n}).mul(-0.0001).add(0.0001)));
        oldParamsTable.put("b", Nd4j.hstack(biases, Nd4j.zeros(biases.rows(), n)));

        MultiLayerNetwork newModel = new TransferLearning.Builder(this.model)
                .nOutReplace(this.model.getLayers().length - 1, weights.columns() + n, WeightInit.ONES)
                .build();

        newModel.getLayer(this.model.getLayers().length - 1).setParamTable(oldParamsTable);
        
        this.model = newModel;
        return this;
    }
    
    public MultiLayerNetwork build() {
        return this.model;
    }
}
