package com.secureai.nn;

import com.secureai.utils.IteratorUtils;
import com.secureai.utils.Nd4jUtils;
import com.secureai.utils.StreamUtils;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.List;
import java.util.Map;

public class DynNNBuilder {

    private MultiLayerNetwork model;
    private int currentLayerIndex = 0; // Default is the first layer
    private int currentLayerBlockSize = 1; // Default is block size of 1

    public DynNNBuilder(MultiLayerNetwork model) {
        this.model = model;
    }

    public DynNNBuilder forLayer(int i) {
        this.currentLayerIndex = i >= 0 ? i : this.model.getLayers().length + i;

        return this;
    }

    public DynNNBuilder setBlockSize(int i) {
        this.currentLayerBlockSize = i;

        return this;
    }

    public int getBlocksCount() {
        return this.model.getLayer(this.currentLayerIndex).getParam("W").columns() / this.currentLayerBlockSize;
    }

    public DynNNBuilder insertOut(int i, int count) {
        Map<String, INDArray> paramsTable = this.model.getLayer(this.currentLayerIndex).paramTable();

        paramsTable.put("W", Nd4jUtils.hInsert(paramsTable.get("W"), Nd4j.rand(paramsTable.get("W").rows(), count * this.currentLayerBlockSize).mul(-0.0001).add(0.0001), i * this.currentLayerBlockSize));
        paramsTable.put("b", Nd4jUtils.hInsert(paramsTable.get("b"), Nd4j.zeros(new int[]{paramsTable.get("b").rows(), count * this.currentLayerBlockSize}), i * this.currentLayerBlockSize));

        return this.setParamTable(paramsTable);
    }

    public DynNNBuilder switchOut(int from, int to) {
        Map<String, INDArray> paramsTable = this.model.getLayer(this.currentLayerIndex).paramTable();

        paramsTable.put("W", Nd4jUtils.hSwitch(paramsTable.get("W"), NDArrayIndex.interval(from * this.currentLayerBlockSize, (from + 1) * this.currentLayerBlockSize), NDArrayIndex.interval(to * this.currentLayerBlockSize, (to + 1) * this.currentLayerBlockSize)));
        paramsTable.put("b", Nd4jUtils.hSwitch(paramsTable.get("b"), NDArrayIndex.interval(from * this.currentLayerBlockSize, (from + 1) * this.currentLayerBlockSize), NDArrayIndex.interval(to * this.currentLayerBlockSize, (to + 1) * this.currentLayerBlockSize)));

        return this.setParamTable(paramsTable);
    }

    public DynNNBuilder deleteOut(int i) {
        Map<String, INDArray> paramsTable = this.model.getLayer(this.currentLayerIndex).paramTable();

        paramsTable.put("W", Nd4jUtils.hDelete(paramsTable.get("W"), NDArrayIndex.interval(i * this.currentLayerBlockSize, 1, (i + 1) * this.currentLayerBlockSize)));
        paramsTable.put("b", Nd4jUtils.hDelete(paramsTable.get("b"), NDArrayIndex.interval(i * this.currentLayerBlockSize, 1, (i + 1) * this.currentLayerBlockSize)));

        return this.setParamTable(paramsTable);
    }

    public DynNNBuilder moveOut(int from, int to) {
        Map<String, INDArray> paramsTable = this.model.getLayer(this.currentLayerIndex).paramTable();

        paramsTable.put("W", Nd4jUtils.hMove(paramsTable.get("W"), NDArrayIndex.interval(from * this.currentLayerBlockSize, (from + 1) * this.currentLayerBlockSize), to * this.currentLayerBlockSize));
        paramsTable.put("b", Nd4jUtils.hMove(paramsTable.get("b"), NDArrayIndex.interval(from * this.currentLayerBlockSize, (from + 1) * this.currentLayerBlockSize), to * this.currentLayerBlockSize));

        return this.setParamTable(paramsTable);
    }

    public DynNNBuilder appendOut(int count) {
        return this.insertOut(this.getBlocksCount(), count);
    }

    public DynNNBuilder transferIn(List<String> oldMap, List<String> newMap) {
        Map<String, INDArray> paramsTable = this.model.getLayer(this.currentLayerIndex).paramTable();

        INDArray weights = paramsTable.get("W");
        INDArray[] newWeights = StreamUtils.fromIterator(IteratorUtils.zipWithIndex(newMap.iterator())).map(
                e -> oldMap.contains(e.getValue()) ?
                        weights.get(NDArrayIndex.interval(oldMap.indexOf(e.getValue()) * this.currentLayerBlockSize, (oldMap.indexOf(e.getValue()) + 1) * this.currentLayerBlockSize)) :
                        Nd4j.rand(this.currentLayerBlockSize, weights.columns()).mul(-0.0001).add(0.0001)
        ).toArray(INDArray[]::new);

        paramsTable.put("W", Nd4j.vstack(newWeights));

        return this.setParamTable(paramsTable);
    }

    public DynNNBuilder transferOut(List<String> oldMap, List<String> newMap) {
        Map<String, INDArray> paramsTable = this.model.getLayer(this.currentLayerIndex).paramTable();

        INDArray weights = paramsTable.get("W");
        INDArray[] newWeights = StreamUtils.fromIterator(IteratorUtils.zipWithIndex(newMap.iterator())).map(
                e -> oldMap.contains(e.getValue()) ?
                        weights.get(NDArrayIndex.all(), NDArrayIndex.interval(oldMap.indexOf(e.getValue()) * this.currentLayerBlockSize, (oldMap.indexOf(e.getValue()) + 1) * this.currentLayerBlockSize)) :
                        Nd4j.rand(weights.rows(), this.currentLayerBlockSize).mul(-0.0001).add(0.0001)
        ).toArray(INDArray[]::new);

        INDArray biases = paramsTable.get("b");
        INDArray[] newBiases = StreamUtils.fromIterator(IteratorUtils.zipWithIndex(newMap.iterator())).map(
                e -> oldMap.contains(e.getValue()) ?
                        biases.get(NDArrayIndex.all(), NDArrayIndex.interval(oldMap.indexOf(e.getValue()) * this.currentLayerBlockSize, (oldMap.indexOf(e.getValue()) + 1) * this.currentLayerBlockSize)) :
                        Nd4j.rand(biases.rows(), this.currentLayerBlockSize).mul(-0.0001).add(0.0001)
        ).toArray(INDArray[]::new);

        paramsTable.put("W", Nd4j.hstack(newWeights));
        paramsTable.put("b", Nd4j.hstack(newBiases));

        return this.setParamTable(paramsTable);
    }

    private DynNNBuilder setParamTable(Map<String, INDArray> paramsTable) {
        MultiLayerNetwork newModel = new TransferLearning.Builder(this.model)
                .nInReplace(this.currentLayerIndex, paramsTable.get("W").rows(), WeightInit.ONES)
                .nOutReplace(this.currentLayerIndex, paramsTable.get("W").columns(), WeightInit.ONES)
                .build();
        newModel.getLayer(this.currentLayerIndex).setParamTable(paramsTable);
        this.model = newModel;

        return this;
    }

    public MultiLayerNetwork build() {
        return this.model;
    }
}
