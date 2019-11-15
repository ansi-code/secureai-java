package com.secureai.rl.abs;

import org.nd4j.linalg.api.ndarray.INDArray;

public interface Environment {
    public void step(Action action);

    public void reset();

    public INDArray getState();

    public Action[] getActions();
}
