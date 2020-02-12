package com.secureai.nn;

import com.secureai.utils.IteratorUtils;
import com.secureai.utils.Nd4jUtils;
import com.secureai.utils.StreamUtils;
import lombok.SneakyThrows;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.List;
import java.util.Map;

public class DynNNBuilder<NN extends MultiLayerNetwork> {

    private MultiLayerNetwork model;
    private int currentLayerIndex = 0; // Default is the first layer
    private int currentLayerBlockSize = 1; // Default is block size of 1

    public DynNNBuilder(NN model) {
        this.model = model;
    }

    public DynNNBuilder<NN> forLayer(int i) {
        this.currentLayerIndex = i >= 0 ? i : this.model.getLayers().length + i;

        return this;
    }

    public DynNNBuilder<NN> setBlockSize(int i) {
        this.currentLayerBlockSize = i;

        return this;
    }

    public int getBlocksCount() {
        return this.model.getLayer(this.currentLayerIndex).getParam("W").columns() / this.currentLayerBlockSize;
    }

    public DynNNBuilder<NN> appendOut(int count) {
        return this.insertOut(this.getBlocksCount(), count);
    }

    public DynNNBuilder<NN> insertOut(int i, int count) {
        Map<String, INDArray> paramsTable = this.model.getLayer(this.currentLayerIndex).paramTable();

        INDArray weights = paramsTable.get("W");
        INDArray biases = paramsTable.get("b");
        paramsTable.put("W", Nd4jUtils.hInsert(weights, Nd4j.rand(weights.rows(), count * this.currentLayerBlockSize).mul(weights.maxNumber().doubleValue() - weights.minNumber().doubleValue()).add(weights.minNumber()), i * this.currentLayerBlockSize));
        paramsTable.put("b", Nd4jUtils.hInsert(biases, Nd4j.rand(biases.rows(), count * this.currentLayerBlockSize).mul(biases.maxNumber().doubleValue() - biases.minNumber().doubleValue()).add(biases.minNumber()), i * this.currentLayerBlockSize));

        return this.setParamTable(paramsTable);
    }

    public DynNNBuilder<NN> switchOut(int from, int to) {
        Map<String, INDArray> paramsTable = this.model.getLayer(this.currentLayerIndex).paramTable();

        paramsTable.put("W", Nd4jUtils.hSwitch(paramsTable.get("W"), NDArrayIndex.interval(from * this.currentLayerBlockSize, (from + 1) * this.currentLayerBlockSize), NDArrayIndex.interval(to * this.currentLayerBlockSize, (to + 1) * this.currentLayerBlockSize)));
        paramsTable.put("b", Nd4jUtils.hSwitch(paramsTable.get("b"), NDArrayIndex.interval(from * this.currentLayerBlockSize, (from + 1) * this.currentLayerBlockSize), NDArrayIndex.interval(to * this.currentLayerBlockSize, (to + 1) * this.currentLayerBlockSize)));

        return this.setParamTable(paramsTable);
    }

    public DynNNBuilder<NN> deleteOut(int i) {
        Map<String, INDArray> paramsTable = this.model.getLayer(this.currentLayerIndex).paramTable();

        paramsTable.put("W", Nd4jUtils.hDelete(paramsTable.get("W"), NDArrayIndex.interval(i * this.currentLayerBlockSize, 1, (i + 1) * this.currentLayerBlockSize)));
        paramsTable.put("b", Nd4jUtils.hDelete(paramsTable.get("b"), NDArrayIndex.interval(i * this.currentLayerBlockSize, 1, (i + 1) * this.currentLayerBlockSize)));

        return this.setParamTable(paramsTable);
    }

    public DynNNBuilder<NN> moveOut(int from, int to) {
        Map<String, INDArray> paramsTable = this.model.getLayer(this.currentLayerIndex).paramTable();

        paramsTable.put("W", Nd4jUtils.hMove(paramsTable.get("W"), NDArrayIndex.interval(from * this.currentLayerBlockSize, (from + 1) * this.currentLayerBlockSize), to * this.currentLayerBlockSize));
        paramsTable.put("b", Nd4jUtils.hMove(paramsTable.get("b"), NDArrayIndex.interval(from * this.currentLayerBlockSize, (from + 1) * this.currentLayerBlockSize), to * this.currentLayerBlockSize));

        return this.setParamTable(paramsTable);
    }

    public DynNNBuilder<NN> transferIn(List<String> oldMap, List<String> newMap) {
        Map<String, INDArray> paramsTable = this.model.getLayer(this.currentLayerIndex).paramTable();

        INDArray weights = paramsTable.get("W");
        INDArray[] newWeights = StreamUtils.fromIterator(IteratorUtils.zipWithIndex(newMap.iterator())).map(
                e -> oldMap.contains(e.getValue()) ?
                        weights.get(NDArrayIndex.interval(oldMap.indexOf(e.getValue()) * this.currentLayerBlockSize, (oldMap.indexOf(e.getValue()) + 1) * this.currentLayerBlockSize)) :
                        Nd4j.rand(this.currentLayerBlockSize, weights.columns()).mul(weights.maxNumber().doubleValue() - weights.minNumber().doubleValue()).add(weights.minNumber()).mul(.1)
        ).toArray(INDArray[]::new);

        paramsTable.put("W", Nd4j.vstack(newWeights));

        return this.setParamTable(paramsTable);
    }

    public DynNNBuilder<NN> transferOut(List<String> oldMap, List<String> newMap) {
        Map<String, INDArray> paramsTable = this.model.getLayer(this.currentLayerIndex).paramTable();

        INDArray weights = paramsTable.get("W");
        INDArray[] newWeights = StreamUtils.fromIterator(IteratorUtils.zipWithIndex(newMap.iterator())).map(
                e -> oldMap.contains(e.getValue()) ?
                        weights.get(NDArrayIndex.all(), NDArrayIndex.interval(oldMap.indexOf(e.getValue()) * this.currentLayerBlockSize, (oldMap.indexOf(e.getValue()) + 1) * this.currentLayerBlockSize)) :
                        Nd4j.rand(weights.rows(), this.currentLayerBlockSize).mul(weights.maxNumber().doubleValue() - weights.minNumber().doubleValue()).add(weights.minNumber()).mul(.1)
        ).toArray(INDArray[]::new);

        INDArray biases = paramsTable.get("b");
        INDArray[] newBiases = StreamUtils.fromIterator(IteratorUtils.zipWithIndex(newMap.iterator())).map(
                e -> oldMap.contains(e.getValue()) ?
                        biases.get(NDArrayIndex.all(), NDArrayIndex.interval(oldMap.indexOf(e.getValue()) * this.currentLayerBlockSize, (oldMap.indexOf(e.getValue()) + 1) * this.currentLayerBlockSize)) :
                        Nd4j.rand(biases.rows(), this.currentLayerBlockSize).mul(biases.maxNumber().doubleValue() - biases.minNumber().doubleValue()).add(biases.minNumber()).mul(.1)
        ).toArray(INDArray[]::new);

        paramsTable.put("W", Nd4j.hstack(newWeights).add(0));
        paramsTable.put("b", Nd4j.hstack(newBiases).add(0));

        return this.setParamTable(paramsTable);
    }

    public DynNNBuilder<NN> replaceIn(List<String> oldMap, List<String> newMap) {
        this.model = new TransferLearning.Builder(this.model)
                .nInReplace(this.currentLayerIndex, newMap.size(), WeightInit.XAVIER)
                .build();

        return this;
    }

    public DynNNBuilder<NN> replaceOut(List<String> oldMap, List<String> newMap) {
        this.model = new TransferLearning.Builder(this.model)
                .nOutReplace(this.currentLayerIndex, newMap.size(), WeightInit.XAVIER)
                .build();

        return this;
    }

    private DynNNBuilder<NN> setParamTable(Map<String, INDArray> paramsTable) {
        MultiLayerNetwork newModel = new TransferLearning.Builder(this.model)
                .nInReplace(this.currentLayerIndex, paramsTable.get("W").rows(), WeightInit.XAVIER)
                .nOutReplace(this.currentLayerIndex, paramsTable.get("W").columns(), WeightInit.XAVIER)
                .build();
        newModel.getLayer(this.currentLayerIndex).setParamTable(paramsTable);
        this.model = newModel;

        return this;
    }

    @SneakyThrows
    public NN build(Class<NN> nnClass) {
        return nnClass.getDeclaredConstructor(MultiLayerConfiguration.class, INDArray.class).newInstance(this.model.getLayerWiseConfigurations(), this.model.params());
    }

    public MultiLayerNetwork build() {
        return this.model;
    }
}
