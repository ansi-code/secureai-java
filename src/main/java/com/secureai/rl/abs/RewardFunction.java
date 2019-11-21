package com.secureai.rl.abs;

public interface RewardFunction<S, A> {
    double reward(S oldState, A action, S currentState);
}
