package com.secureai.rl.rl4j;

import com.secureai.model.Topology;
import com.secureai.rl.rl4j.abs.DiscreteState;
import com.secureai.utils.RandomUtils;

public class SystemState extends DiscreteState {

    public SystemState(Topology topology) {
        super(topology.getNodes().size(), NodeState.values().length);
    }

    @Override
    public void reset() {
        super.reset();

        for (int i = 0; i < this.size(); i++) {
            this.set(i, NodeState.active, RandomUtils.getRandom().nextDouble() < 0.7);
            this.set(i, NodeState.updated, RandomUtils.getRandom().nextDouble() < 0.5);
            this.set(i, NodeState.corrupted, RandomUtils.getRandom().nextDouble() > 0.6);
            this.set(i, NodeState.vulnerable, RandomUtils.getRandom().nextDouble() > 0.7);
        }
    }

    public void worst() {
        super.reset();

        for (int i = 0; i < this.size(); i++) {
            this.set(i, NodeState.active, false);
            this.set(i, NodeState.updated, false);
            this.set(i, NodeState.corrupted, true);
            this.set(i, NodeState.vulnerable, true);
        }
    }

    public boolean get(int i, NodeState nodeState) {
        return this.get(i, nodeState.getValue()) == 1;
    }

    public SystemState set(int i, NodeState nodeState, boolean value) {
        this.set(i, nodeState.getValue(), value ? 1 : 0);
        return this;
    }

    public long size() {
        return this.getState().shape()[0]; // nodes count
    }
}
