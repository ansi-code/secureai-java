package com.secureai.rl.abs;

public interface TerminateFunction<S> {
    boolean terminated(S state);
}
