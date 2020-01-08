package com.secureai.nn;

import com.secureai.utils.Nd4jUtils;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.Map;

public class DynNNBuilder {

    private MultiLayerNetwork model;
    private int currentLayerIndex = 0; // Default is the first layer
    private int currentLayerBlockSize = 1; // Default is block size of 1

    public DynNNBuilder(MultiLayerNetwork model) {
        this.model = model;
    }

    public DynNNBuilder forLayer(int i) {
        this.currentLayerIndex = i > 0 ? i : this.model.getLayers().length + i;

        return this;
    }

    public DynNNBuilder setBlockSize(int i) {
        this.currentLayerBlockSize = i;

        return this;
    }

    public int getBlocksCount() {
        return this.model.getLayer(this.currentLayerIndex).getParam("W").columns() / this.currentLayerBlockSize;
    }

    public DynNNBuilder insertOutputBlock(int i) {
        Map<String, INDArray> paramsTable = this.model.getLayer(this.currentLayerIndex).paramTable();
        INDArray weights = paramsTable.get("W");
        INDArray biases = paramsTable.get("b");

        paramsTable.put("W", Nd4jUtils.hInsert(weights, Nd4j.rand(new int[]{weights.rows(), this.currentLayerBlockSize}).mul(-0.0001).add(0.0001), i * this.currentLayerBlockSize));
        paramsTable.put("b", Nd4jUtils.hInsert(biases, Nd4j.zeros(new int[]{biases.rows(), this.currentLayerBlockSize}), i * this.currentLayerBlockSize));

        MultiLayerNetwork newModel = new TransferLearning.Builder(this.model)
                .nOutReplace(this.currentLayerIndex, weights.columns() + this.currentLayerBlockSize, WeightInit.ONES)
                .build();
        newModel.getLayer(this.currentLayerIndex).setParamTable(paramsTable);
        this.model = newModel;

        return this;
    }

    public DynNNBuilder switchOutputBlock(int from, int to) {
        Map<String, INDArray> paramsTable = this.model.getLayer(this.currentLayerIndex).paramTable();

        paramsTable.put("W", Nd4jUtils.hSwitch(paramsTable.get("W"), NDArrayIndex.interval(from * this.currentLayerBlockSize, (from + 1) * this.currentLayerBlockSize), NDArrayIndex.interval(to * this.currentLayerBlockSize, (to + 1) * this.currentLayerBlockSize)));
        paramsTable.put("b", Nd4jUtils.hSwitch(paramsTable.get("b"), NDArrayIndex.interval(from * this.currentLayerBlockSize, (from + 1) * this.currentLayerBlockSize), NDArrayIndex.interval(to * this.currentLayerBlockSize, (to + 1) * this.currentLayerBlockSize)));

        this.model.getLayer(this.currentLayerIndex).setParamTable(paramsTable);

        return this;
    }

    public DynNNBuilder deleteOutputBlock(int i) {
        Map<String, INDArray> paramsTable = this.model.getLayer(this.currentLayerIndex).paramTable();

        paramsTable.put("W", Nd4jUtils.hDelete(paramsTable.get("W"), NDArrayIndex.interval(i * this.currentLayerBlockSize, 1, (i + 1) * this.currentLayerBlockSize)));
        paramsTable.put("b", Nd4jUtils.hDelete(paramsTable.get("b"), NDArrayIndex.interval(i * this.currentLayerBlockSize, 1, (i + 1) * this.currentLayerBlockSize)));

        this.model.getLayer(this.currentLayerIndex).setParamTable(paramsTable);

        return this;
    }

    public DynNNBuilder moveOutputBlock(int from, int to) {
        Map<String, INDArray> paramsTable = this.model.getLayer(this.currentLayerIndex).paramTable();

        paramsTable.put("W", Nd4jUtils.hMove(paramsTable.get("W"), NDArrayIndex.interval(from * this.currentLayerBlockSize, (from + 1) * this.currentLayerBlockSize), to * this.currentLayerBlockSize));
        paramsTable.put("b", Nd4jUtils.hMove(paramsTable.get("b"), NDArrayIndex.interval(from * this.currentLayerBlockSize, (from + 1) * this.currentLayerBlockSize), to * this.currentLayerBlockSize));

        this.model.getLayer(this.currentLayerIndex).setParamTable(paramsTable);

        return this;
    }

    public DynNNBuilder appendOutputBlock() {
        return this.insertOutputBlock(this.getBlocksCount());
    }

    public MultiLayerNetwork build() {
        return this.model;
    }
}
