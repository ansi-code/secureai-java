package com.secureai.system;

import com.secureai.Config;
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
        if (oldState.equals(currentState)) return -2;
        Action action = this.environment.getActionSet().getActions().get(systemAction.getActionId());
        return -(Config.TIME_WEIGHT * (action.getExecutionTime() / this.maxExecutionTime) + Config.COST_WEIGHT * (action.getExecutionCost() / this.maxExecutionCost));
    }

}
