package com.secureai.system;

import com.secureai.Config;
import com.secureai.model.actionset.ActionSet;
import com.secureai.model.topology.Topology;
import com.secureai.rl.abs.SMDP;
import com.secureai.utils.MapCounter;
import com.secureai.utils.Stat;
import lombok.Getter;
import org.deeplearning4j.gym.StepReply;
import org.deeplearning4j.rl4j.space.DiscreteSpace;
import org.json.JSONObject;

public class SystemEnvironment implements SMDP<SystemState, Integer, DiscreteSpace> {

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
    @Getter
    private int episodes = 0;
    @Getter
    private double cumulativeReward = 0;

    private MapCounter<String> actionCounter;
    private Stat<Double> stat;

    public SystemEnvironment(Topology topology, ActionSet actionSet) {
        this.actionSet = actionSet;

        this.systemDefinition = new SystemDefinition(topology);
        this.actionSpace = new SystemActionSpace(this);
        this.observationSpace = new SystemStateSpace(this);
        this.systemState = new SystemState(this);
        this.systemRewardFunction = new SystemRewardFunction(this);
        this.systemTerminateFunction = new SystemTerminateFunction(this);
        this.actionCounter = new MapCounter<>();
        this.stat = new Stat<>("output/mdp");
    }

    public void close() {
        this.stat.flush();
    }

    public boolean isDone() {
        return systemTerminateFunction.terminated(this.systemState) || this.step >= Config.MAX_STEPS;
    }

    public SystemState reset() {
        this.systemState.reset();
        this.step = 0;
        this.episodes++;
        this.actionCounter = new MapCounter<>();
        this.cumulativeReward = 0;
        this.stat.flush();
        return this.systemState;
    }

    public StepReply<SystemState> step(Integer a) {
        this.step++;

        SystemState oldState = this.systemState;
        SystemAction action = this.actionSpace.encode(a);
        action.run(this);
        SystemState currentState = this.systemState;

        double reward = systemRewardFunction.reward(oldState, action, currentState);
        boolean done = this.isDone();
        this.actionCounter.increment(String.format("%s-%s", action.getResourceId(), action.getActionId()));
        this.cumulativeReward += reward;
        if (done) {
            this.stat.append(this.cumulativeReward);
            System.out.println(String.format("[Episode %d][Steps: %d][Cumulative Reward: %f][Action bins %s]", this.episodes, this.step, this.cumulativeReward, this.actionCounter));
        }
        //System.out.println("ACTION STEP: " + a + "; REWARD: " + reward); //nsccf

        return new StepReply<>(currentState, reward, isDone(), new JSONObject("{}"));
    }

    public SystemEnvironment newInstance() {
        return new SystemEnvironment(this.systemDefinition.getTopology(), this.actionSet);
    }

    @Override
    public SystemState getState() {
        return this.systemState;
    }

    @Override
    public void setState(SystemState state) {
        this.systemState = state;
    }
}
