package com.secureai.rl.qn;


import com.secureai.utils.ArrayUtils;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

public class DynamicQTable implements QTable {

    private final int actionSpaceSize;
    private Map<Integer, double[]> map;

    public DynamicQTable(int actionSpaceSize) {
        this.actionSpaceSize = actionSpaceSize;
        this.map = new HashMap<>();
    }

    private double[] getOrPut(int state) {
        if (!this.map.containsKey(state))
            this.map.put(state, new double[this.actionSpaceSize]);
        return this.map.get(state);
    }

    @Override
    public double get(int state, int action) {
        return this.getOrPut(state)[action];
    }

    @Override
    public void set(int state, int action, double value) {
        this.getOrPut(state)[action] = value;
    }

    @Override
    public double[] get(int state) {
        return this.getOrPut(state);
    }

    @Override
    public double max(int state) {
        return Arrays.stream(this.getOrPut(state)).max().orElse(0d);
    }

    @Override
    public int argMax(int state) {
        return ArrayUtils.argmax(this.getOrPut(state));
    }
}