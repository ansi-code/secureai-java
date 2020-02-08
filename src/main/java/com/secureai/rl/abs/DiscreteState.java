package com.secureai.rl.abs;

import com.secureai.utils.ArrayUtils;
import lombok.Getter;
import lombok.Setter;
import org.deeplearning4j.rl4j.space.Encodable;

import java.util.Arrays;

public class DiscreteState implements Encodable {

    @Getter
    @Setter
    private double[] state;

    public DiscreteState(double[] state) {
        this.state = state;
    }

    public DiscreteState(int size) {
        this.state = new double[size];
    }

    public double get(int index) {
        return this.state[index];
    }

    public void set(int v, int index) {
        this.state[index] = v;
    }

    public void reset() {
        this.state = new double[this.state.length];
    }

    @Override
    public double[] toArray() {
        return this.state;
    }

    public int[] toIntArray() {
        return Arrays.stream(this.state).mapToInt(v -> (int) v).toArray();
    }

    public int toInt() {
        return ArrayUtils.toBase10(this.toIntArray(), 2);
    }

    public DiscreteState setFromInt(int value) {
        this.state = this.fromInt(value);
        return this;
    }

    public DiscreteState newInstance() {
        return new DiscreteState(this.state.length);
    }

    public DiscreteState newInstance(int value) {
        DiscreteState result = new DiscreteState(this.state.length);
        result.setFromInt(value);
        return result;
    }

    public double[] fromInt(int value) {
        int[] data = ArrayUtils.fromBase10(value, 2);
        double[] result = new double[this.state.length];
        for (int i = 0; i < data.length; i++)
            result[result.length - i - 1] = data[data.length - i - 1];
        return result;
    }

    public boolean equals(DiscreteState other) {
        return Arrays.equals(state, other.state);
    }
}
