package com.secureai.rl.abs;

import org.deeplearning4j.rl4j.mdp.MDP;
import org.deeplearning4j.rl4j.space.ActionSpace;

public interface SMDP<O, A, AS extends ActionSpace<A>> extends MDP<O, A, AS> {
    O getState();

    void setState(O state);
}
