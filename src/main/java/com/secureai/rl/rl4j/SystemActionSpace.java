package com.secureai.rl.rl4j;

import com.secureai.model.Topology;
import lombok.Getter;
import org.deeplearning4j.rl4j.space.DiscreteSpace;

public class SystemActionSpace extends DiscreteSpace {

    @Getter
    private double maxExecutionTime;
    @Getter
    private double maxExecutionCost;

    public SystemActionSpace(Topology topology) {
        super(topology.getNodes().size() * NodeAction.values().length);
    }

    @Override
    public SystemAction encode(Integer a) {
        return new SystemAction(a / NodeAction.values().length, NodeAction.values()[a % NodeAction.values().length]);
    }

    public int size() {
        return this.size;
    }
}
