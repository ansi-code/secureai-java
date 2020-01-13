package com.secureai.system;

import com.secureai.rl.abs.ArrayObservationSpace;

public class SystemStateSpace extends ArrayObservationSpace<SystemState> {

    private SystemDefinition systemDefinition;

    public SystemStateSpace(SystemDefinition systemDefinition) {
        super(new int[]{systemDefinition.getTopology().getResources().size() * NodeState.values().length});
        this.systemDefinition = systemDefinition;
    }

    public long getReplications(Integer nodeIndex) {
        return this.systemDefinition.getNode(nodeIndex).getReplication();
    }

    public long getInConnections(Integer nodeIndex) {
        return this.systemDefinition.getInConnectionsCount(this.systemDefinition.getNodeName(nodeIndex));
    }

    public long getOutConnections(Integer nodeIndex) {
        return this.systemDefinition.getOutConnectionsCount(this.systemDefinition.getNodeName(nodeIndex));
    }

    public int size() {
        return this.systemDefinition.getTopology().getResources().size() * NodeState.values().length;
    }

}
