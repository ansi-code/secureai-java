package com.secureai.rl;

import com.secureai.model.Topology;
import com.secureai.rl.abs.Action;
import com.secureai.rl.abs.Environment;
import org.nd4j.linalg.api.ndarray.INDArray;

public class SystemEnvironment implements Environment {

    private SystemState state;

    public SystemEnvironment(Topology topology) {
        this.state = new SystemState(topology);
    }

    @Override
    public void step(Action action) {

    }

    @Override
    public void reset() {

    }

    @Override
    public INDArray getState() {
        return this.state.getArray();
    }

    @Override
    public Action[] getActions() {
        return new Action[0];
    }
}
