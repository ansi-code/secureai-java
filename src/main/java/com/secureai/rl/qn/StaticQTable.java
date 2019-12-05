package com.secureai.rl.qn;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class StaticQTable implements QTable {

    private INDArray array;

    public StaticQTable(int stateSpaceSize, int actionSpaceSize) {
        this.array = Nd4j.rand(stateSpaceSize, actionSpaceSize);
    }

    @Override
    public double get(int state, int action) {
        return this.array.getDouble(state, action);
    }

    @Override
    public void set(int state, int action, double value) {
        this.array.put(state, action, value);
    }

    @Override
    public double[] get(int state) {
        return this.array.getRow(state).toDoubleVector();
    }

    @Override
    public double max(int state) {
        return this.array.getRow(state).maxNumber().doubleValue();
    }

    @Override
    public int argMax(int state) {
        return this.array.getRow(state).argMax().getInt(0);
    }
}
