package com.secureai.rl.news;

import com.secureai.rl.news.abs.RewardFunction;

public class SystemRewardFunction implements RewardFunction<SystemState, SystemAction> {
    @Override
    public double reward(SystemState oldState, SystemAction action, SystemState currentState) {
        return 0;
    }
}
