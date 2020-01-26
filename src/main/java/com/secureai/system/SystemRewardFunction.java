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
    public double reward(SystemState oldState, SystemAction systemAction, SystemState currentState) {
        Action action = this.environment.getActionSet().getActions().get(systemAction.getActionId());
        return -((action.getExecutionTime() / this.maxExecutionTime) + (action.getExecutionCost() / this.maxExecutionCost)) * this.destruction(systemAction, currentState);
    }

    public double destruction(SystemAction systemAction, SystemState currentState) {
        Action action = this.environment.getActionSet().getActions().get(systemAction.getActionId());
        if (!action.getDisruptive())
            return 1d / environment.getSystemDefinition().getResources().size();

        if (this.environment.getSystemDefinition().getTask(systemAction.getResourceId()).getReplication() > 1)
            return 1d / environment.getSystemDefinition().getResources().size();

        return ((double) this.environment.getSystemDefinition().getInConnectionsCount(systemAction.getResourceId()) + this.environment.getSystemDefinition().getOutConnectionsCount(systemAction.getResourceId())) / environment.getSystemDefinition().getResources().size();
    }

}
