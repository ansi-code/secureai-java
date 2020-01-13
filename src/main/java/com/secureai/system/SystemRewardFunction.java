package com.secureai.system;

import com.secureai.rl.abs.RewardFunction;

public class SystemRewardFunction implements RewardFunction<SystemState, SystemAction> {

    private SystemActionSpace actionSpace;
    private SystemStateSpace stateSpace;

    public SystemRewardFunction(SystemActionSpace actionSpace, SystemStateSpace stateSpace) {
        this.actionSpace = actionSpace;
        this.stateSpace = stateSpace;
    }

    @Override
    public double reward(SystemState oldState, SystemAction action, SystemState currentState) {
        return -((action.getAction().getExecutionTime() / this.actionSpace.getMaxExecutionTime()) + (action.getAction().getExecutionCost() / this.actionSpace.getMaxExecutionCost())) * this.destruction(action, currentState);
    }

    public double destruction(SystemAction action, SystemState currentState) {
        if (!action.getAction().getDisruptive())
            return 1d / currentState.size();

        if (this.stateSpace.getReplications(action.getNodeIndex()) > 1)
            return 1d / currentState.size();

        return ((double) this.stateSpace.getInConnections(action.getNodeIndex()) + this.stateSpace.getOutConnections(action.getNodeIndex())) / currentState.size();
    }

}
