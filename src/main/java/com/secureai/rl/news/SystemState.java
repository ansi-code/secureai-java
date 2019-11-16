package com.secureai.rl.news;

import lombok.Getter;
import org.deeplearning4j.rl4j.space.Encodable;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class SystemState implements Encodable {

    @Getter
    private INDArray state;

    public SystemState(double[] state) {
        this.state = Nd4j.create(state);
    }

    public SystemState(int... shape) {
        this.state = Nd4j.rand(shape);
    }

    public double get(int... shape) {
        return this.state.getDouble(shape);
    }

    public void set(double v, int... shape) {
        this.state.putScalar(shape, v);
    }

    public void reset() {
        this.state = Nd4j.rand(this.state.shape());
    }

    @Override
    public double[] toArray() {
        return this.state.toDoubleVector();
    }

}
