package com.secureai.rl.news;

import com.secureai.model.Topology;
import com.secureai.rl.news.abs.ArrayObservationSpace;

public class SystemStateSpace extends ArrayObservationSpace<SystemState> {

    public SystemStateSpace(Topology topology) {
        super(new int[]{topology.getNodes().size() * NodeState.values().length});
    }

    public enum NodeState {
        active, updated, updatable, corrupted, vulnerable
    }
}
