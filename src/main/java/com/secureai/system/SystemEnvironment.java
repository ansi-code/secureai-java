package com.secureai.system;

import com.secureai.Config;
import com.secureai.model.actionset.ActionSet;
import com.secureai.model.topology.Topology;
import lombok.Getter;
import org.deeplearning4j.gym.StepReply;
import org.deeplearning4j.rl4j.mdp.MDP;
import org.deeplearning4j.rl4j.space.DiscreteSpace;
import org.json.JSONObject;

public class SystemEnvironment implements MDP<SystemState, Integer, DiscreteSpace> { // SystemActionSpace

    @Getter
    private SystemActionSpace actionSpace;
    @Getter
    private ActionSet actionSet;
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
    private SystemDefinition systemDefinition;

    public SystemEnvironment(Topology topology, ActionSet actionSet) {
        this.systemDefinition = new SystemDefinition(topology);

        this.actionSpace = new SystemActionSpace(this.systemDefinition, actionSet);
        this.observationSpace = new SystemStateSpace(this.systemDefinition);
        this.systemState = new SystemState(this.systemDefinition);
        this.systemRewardFunction = new SystemRewardFunction(this.actionSpace, this.observationSpace);
        this.systemTerminateFunction = new SystemTerminateFunction();

        this.step = 0;
        this.topology = topology;
        this.actionSet = actionSet;
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
        return new SystemEnvironment(this.topology, this.actionSet);
    }

}
