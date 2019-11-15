package com.secureai.rl;

import com.secureai.model.Topology;
import lombok.Getter;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class SystemState {

    @Getter
    private INDArray array;

    public SystemState(Topology topology) {
        this.array = Nd4j.zeros(topology.getNodes().size() * NodeStates.values().length);
    }

    public enum NodeStates {
        active, updated, updatable, corrupted, vulnerable
    }
}
