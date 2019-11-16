package com.secureai.rl.news;

import lombok.Getter;
import org.deeplearning4j.rl4j.space.Encodable;

import java.util.Arrays;

public class SystemState implements Encodable {

    @Getter
    private double[] state; // Unrolled state

    public SystemState(double[] state) {
        this.state = state;
    }

    public SystemState(int... shape) {
        this.state = new double[Arrays.stream(shape).reduce(1, (left, right) -> left * right)];
    }

    public double get(int... shape) {
        return this.state[Arrays.stream(shape).reduce(1, (left, right) -> left * right)]; //TODO
    }

    public void set(double v, int... shape) {
        this.state[Arrays.stream(shape).reduce(1, (left, right) -> left * right)] = v; //TODO
    }

    @Override
    public double[] toArray() {
        return state;
    }

    public int to1D( int x, int y, int z ) {
        return (z * xMax * yMax) + (y * xMax) + x;
    }

    public int[] to3D( int idx ) {
        final int z = idx / (xMax * yMax);
        idx -= (z * xMax * yMax);
        final int y = idx / xMax;
        final int x = idx % xMax;
        return new int[]{ x, y, z };
    }

}
