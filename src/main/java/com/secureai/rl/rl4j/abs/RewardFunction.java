package com.secureai.rl.rl4j.abs;

public interface RewardFunction<S, A> {
    double reward(S oldState, A action, S currentState);
}
