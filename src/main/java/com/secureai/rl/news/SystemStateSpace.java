package com.secureai.rl.news;

import com.secureai.model.Topology;
import org.nd4j.linalg.factory.Nd4j;

public class SystemStateSpace {

    public SystemStateSpace(Topology topology) {
        this.state = Nd4j.zeros(topology.getNodes().size() * NodeState.values().length);
    }

    public enum NodeState {
        active, updated, updatable, corrupted, vulnerable
    }
}
