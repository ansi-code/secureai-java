package com.secureai.rl.abs;

import com.secureai.utils.ArrayUtils;
import com.secureai.utils.Nd4jUtils;
import lombok.Getter;
import lombok.Setter;
import org.deeplearning4j.rl4j.space.Encodable;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class DiscreteState implements Encodable {

    @Getter
    @Setter
    private INDArray state;

    public DiscreteState(INDArray state) {
        this.state = state;
    }

    public DiscreteState(long... shape) {
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
        return this.state.ravel().toDoubleVector(); // It can use the space decode function
    }

    public int toInt() {
        return ArrayUtils.toBase10(this.state.ravel().toIntVector(), 2);
    }

    public DiscreteState setFromInt(int value) {
        this.state = this.fromInt(value);
        return this;
    }

    public DiscreteState newInstance() {
        return new DiscreteState(this.state.shape());
    }

    public DiscreteState newInstance(int value) {
        DiscreteState result = new DiscreteState(this.state.shape());
        result.setFromInt(value);
        return result;
    }

    public INDArray fromInt(int value) {
        INDArray result = Nd4j.zeros(ArrayUtils.multiply(this.state.shape()));
        int[] data = ArrayUtils.fromBase10(value, 2);
        INDArray base2 = Nd4j.create(data, new long[]{data.length}, DataType.INT);
        result.put(Nd4jUtils.createRightCoveringShape(base2.shape(), result.shape()), base2);
        return result.reshape(this.state.shape());
    }

    @Override
    public boolean equals(Object obj) {
        return this.state.equalsWithEps(obj, 1);
    }

    public INDArray cloneState() {
        return this.state.add(0);
    }
}
