package com.secureai.system;

import com.secureai.Config;
import com.secureai.model.actionset.ActionSet;
import com.secureai.model.topology.Topology;
import lombok.Getter;
import org.deeplearning4j.gym.StepReply;
import org.deeplearning4j.rl4j.mdp.MDP;
import org.deeplearning4j.rl4j.space.DiscreteSpace;
import org.json.JSONObject;

public class SystemEnvironment implements MDP<SystemState, Integer, DiscreteSpace> {

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
    private SystemDefinition systemDefinition;

    @Getter
    private int step = 0;

    public SystemEnvironment(Topology topology, ActionSet actionSet) {
        this.actionSet = actionSet;

        this.systemDefinition = new SystemDefinition(topology);
        this.actionSpace = new SystemActionSpace(this);
        this.observationSpace = new SystemStateSpace(this);
        this.systemState = new SystemState(this);
        this.systemRewardFunction = new SystemRewardFunction(this);
        this.systemTerminateFunction = new SystemTerminateFunction(this);
    }

    public void close() {

    }

    public boolean isDone() {
        /*
        if (systemTerminateFunction.terminated(this.systemState) || this.step >= Config.MAX_STEPS)
            System.out.println("EPISODE END");
         */
        return systemTerminateFunction.terminated(this.systemState) || this.step >= Config.MAX_STEPS;
    }

    public SystemState reset() {
        this.systemState.reset();
        this.step = 0;
        return this.systemState;
    }

    public StepReply<SystemState> step(Integer a) {
        //System.out.println("ACTION STEP: " + a);
        this.step++;

        SystemState oldState = this.systemState;
        SystemAction action = this.actionSpace.encode(a);
        action.run(this);
        SystemState currentState = this.systemState;

        double reward = systemRewardFunction.reward(oldState, action, currentState);

        return new StepReply<>(currentState, reward, isDone(), new JSONObject("{}"));
    }

    public SystemEnvironment newInstance() {
        return new SystemEnvironment(this.systemDefinition.getTopology(), this.actionSet);
    }

}
