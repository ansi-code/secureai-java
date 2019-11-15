package com.secureai.rl.rl4j;

import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.gym.StepReply;
import org.deeplearning4j.rl4j.mdp.MDP;
import org.deeplearning4j.rl4j.space.ArrayObservationSpace;
import org.deeplearning4j.rl4j.space.DiscreteSpace;
import org.deeplearning4j.rl4j.space.ObservationSpace;
import org.json.JSONObject;

@Slf4j
public class SystemEnv implements MDP<SystemState, Integer, DiscreteSpace> {
    @Getter
    private DiscreteSpace actionSpace = new DiscreteSpace(2);
    @Getter
    private ObservationSpace<SystemState> observationSpace = new ArrayObservationSpace<>(new int[]{1});
    private SystemState systemState;

    public void close() {
    }

    @Override
    public boolean isDone() {
        return systemState.getStep() == 10;
    }

    public SystemState reset() {
        return systemState = new SystemState(0, 0);
    }

    public StepReply<SystemState> step(Integer a) {
        double reward = (systemState.getStep() % 2 == 0) ? 1 - a : a;
        systemState = new SystemState(systemState.getI() + 1, systemState.getStep() + 1);
        return new StepReply<>(systemState, reward, isDone(), new JSONObject("{}"));
    }

    public SystemEnv newInstance() {
        return new SystemEnv();
    }

}
