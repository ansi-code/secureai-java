package com.secureai.rl.rl4j.abs;

public interface TerminateFunction<S> {
    boolean terminated(S state);
}
