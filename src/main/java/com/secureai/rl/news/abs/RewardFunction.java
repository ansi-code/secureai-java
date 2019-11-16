package com.secureai.rl.news.abs;

public interface RewardFunction<S, A> {
    double reward(S oldState, A action, S currentState);
}
