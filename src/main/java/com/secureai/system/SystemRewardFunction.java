package com.secureai.system;

import com.secureai.model.actionset.Action;
import com.secureai.rl.abs.RewardFunction;
import lombok.Getter;

public class SystemRewardFunction implements RewardFunction<SystemState, SystemAction> {

    private SystemEnvironment environment;

    @Getter
    private double maxExecutionTime;
    @Getter
    private double maxExecutionCost;

    public SystemRewardFunction(SystemEnvironment environment) {
        this.environment = environment;

        this.maxExecutionTime = this.environment.getActionSet().getActions().values().stream().map(Action::getExecutionTime).max(Double::compareTo).orElse(0d);
        this.maxExecutionCost = this.environment.getActionSet().getActions().values().stream().map(Action::getExecutionCost).max(Double::compareTo).orElse(0d);
    }

    @Override
    public double reward(SystemState oldState, SystemAction action, SystemState currentState) {
        return -((action.getAction().getExecutionTime() / this.maxExecutionTime) + (action.getAction().getExecutionCost() / this.maxExecutionCost)) * this.destruction(action, currentState);
    }

    public double destruction(SystemAction action, SystemState currentState) {
        if (!action.getAction().getDisruptive())
            return 1d / environment.getSystemDefinition().getTopology().getResources().size();

        if (this.environment.getSystemDefinition().getTopology().getResources().get(action.getResourceId()).getReplication() > 1)
            return 1d / environment.getSystemDefinition().getTopology().getResources().size();

        return ((double) this.environment.getSystemDefinition().getInConnectionsCount(action.getResourceId()) + this.environment.getSystemDefinition().getOutConnectionsCount(action.getResourceId())) / environment.getSystemDefinition().getTopology().getResources().size();
    }

}
