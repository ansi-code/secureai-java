package com.secureai.rl.abs;

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

    private static int toBase10(int[] values, int base) {
        int num = 0;
        for (int i = values.length - 1, power = 1; i >= 0; power *= base)
            num += values[i--] * power;
        return num;
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

    public int toInt() {
        return toBase10(this.state.ravel().toIntVector(), 2);
    }

}
