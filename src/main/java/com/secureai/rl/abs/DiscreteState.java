package com.secureai.rl.abs;

import com.secureai.utils.ArrayUtils;
import lombok.Getter;
import org.deeplearning4j.rl4j.space.Encodable;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

public class DiscreteState implements Encodable {

    @Getter
    private INDArray state;

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

    public void setFromInt(int value) {
        this.state = this.fromInt(value);
    }

    public DiscreteState newInstance() {
        return new DiscreteState(this.state.shape());
    }

    public INDArray fromInt(int value) {
        INDArray result = Nd4j.zeros(ArrayUtils.multiply(this.state.shape()));
        int[] data = ArrayUtils.fromBase10(value, 2);
        INDArray base2 = Nd4j.create(data, new long[]{data.length}, DataType.INT);
        result.put(NDArrayIndex.createCoveringShape(base2.shape()), base2);
        return result.reshape(this.state.shape());
    }

}
