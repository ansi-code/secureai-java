package com.secureai.rl.qn;

public interface QTable {
    double get(int state, int action);

    void set(int state, int action, double value);

    double[] get(int state);

    double max(int state);

    int argMax(int state);
}
