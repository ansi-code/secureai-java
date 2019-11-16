package com.secureai.rl.rl4j;

import com.secureai.Config;
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
    private SystemRewardFunction systemRewardFunction;
    @Getter
    private SystemTerminateFunction systemTerminateFunction;
    @Getter
    private int step;

    private Topology topology;

    public SystemEnvironment(Topology topology) {
        this.actionSpace = new SystemActionSpace(topology);
        this.observationSpace = new SystemStateSpace(topology);
        this.systemState = new SystemState(topology);
        this.systemRewardFunction = new SystemRewardFunction(this.actionSpace, this.observationSpace);
        this.systemTerminateFunction = new SystemTerminateFunction();
        this.step = 0;
        this.topology = topology;
    }

    public void close() {

    }

    public boolean isDone() {
        return systemTerminateFunction.terminated(this.systemState) || this.step >= Config.MAX_STEPS;
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
