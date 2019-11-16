package com.secureai.rl.news;

import com.secureai.model.Topology;
import lombok.Getter;
import org.deeplearning4j.gym.StepReply;
import org.deeplearning4j.rl4j.mdp.MDP;
import org.json.JSONObject;

public class SystemEnvironment implements MDP<SystemState, Integer, SystemActionSpace> {
    @Getter
    private SystemActionSpace actionSpace;
    @Getter
    private SystemStateSpace observationSpace;
    @Getter
    private SystemState systemState;
    @Getter
    private SystemRewardFunction systemRewardFunction = new SystemRewardFunction();
    @Getter
    private int step;

    private Topology topology;

    public SystemEnvironment(Topology topology) {
        this.actionSpace = new SystemActionSpace(topology);
        this.observationSpace = new SystemStateSpace(topology);
        this.systemState = new SystemState();
        this.step = 0;
        this.topology = topology;
    }

    public void close() {

    }

    public boolean isDone() {
        return this.step == 100;
    }

    public SystemState reset() {
        this.systemState.reset();
        this.step = 0;
        return this.systemState;
    }

    public StepReply<SystemState> step(Integer a) {
        this.step++;

        SystemState oldState = this.systemState;
        SystemAction action = this.actionSpace.encode(a);
        action.run(this.systemState);
        SystemState currentState = this.systemState;

        double reward = systemRewardFunction.reward(oldState, action, currentState);

        return new StepReply<>(currentState, reward, isDone(), new JSONObject("{}"));
    }

    public SystemEnvironment newInstance() {
        return new SystemEnvironment(this.topology);
    }

}
