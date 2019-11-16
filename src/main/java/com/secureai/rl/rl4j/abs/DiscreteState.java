package com.secureai.rl.rl4j.abs;

import lombok.Getter;
import org.deeplearning4j.rl4j.space.Encodable;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class DiscreteState implements Encodable {

    @Getter
    private INDArray state;

    public DiscreteState(int... shape) {
        this.state = Nd4j.zeros(shape);
    }

    public int get(int... indices) {
        return this.state.getInt(indices);
    }

    public int[] getRow(int... indices) {
        return this.state.getRows(indices).toIntVector();
    }

    public void set(int v, int... indices) {
        this.state.putScalar(indices, v);
    }

    public void reset() {
        this.state = Nd4j.zeros(this.state.shape());
    }

    @Override
    public double[] toArray() {
        return this.state.ravel().toDoubleVector();
    }

}
