package com.secureai.system;

import com.secureai.model.Topology;
import com.secureai.rl.abs.ArrayObservationSpace;

public class SystemStateSpace extends ArrayObservationSpace<SystemState> {

    private Topology topology;

    public SystemStateSpace(Topology topology) {
        super(new int[]{topology.getNodes().size() * NodeState.values().length});
        this.topology = topology;
    }

    public long getReplications(Integer nodeIndex) {
        return this.topology.getNode(nodeIndex).getReplication();
    }

    public long getInConnections(Integer nodeIndex) {
        return this.topology.getInEdgesCount(this.topology.getNodeName(nodeIndex));
    }

    public long getOutConnections(Integer nodeIndex) {
        return this.topology.getOutEdgesCount(this.topology.getNodeName(nodeIndex));
    }

    public int size() {
        return topology.getNodes().size() * NodeState.values().length;
    }

}
