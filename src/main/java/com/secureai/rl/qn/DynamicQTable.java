package com.secureai.rl.qn;


import com.secureai.utils.ArrayUtils;
import com.secureai.utils.JSONUtils;

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
        return this.get(state)[action];
    }

    @Override
    public void set(int state, int action, double value) {
        this.get(state)[action] = value;
    }

    @Override
    public double[] get(int state) {
        return this.getOrPut(state);
    }

    @Override
    public double max(int state) {
        return Arrays.stream(this.get(state)).max().orElse(0d);
    }

    @Override
    public int argMax(int state) {
        return ArrayUtils.argMax(this.get(state));
    }

    @Override
    public String toString() {
        return "DynamicQTable{" +
                "actionSpaceSize=" + this.actionSpaceSize +
                ", map=" + JSONUtils.toJSON(this.map) +
                '}';
    }
}
