package com.secureai.rl.vi;

import com.secureai.rl.abs.DiscreteState;
import com.secureai.utils.RandomUtils;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import org.deeplearning4j.gym.StepReply;
import org.deeplearning4j.rl4j.mdp.MDP;
import org.deeplearning4j.rl4j.space.DiscreteSpace;

import java.util.logging.Logger;

public class ValueIteration<O extends DiscreteState> {

    private final static Logger LOGGER = Logger.getLogger(ValueIteration.class.getName());

    private ValueIteration.VIConfiguration conf;
    private MDP<O, Integer, DiscreteSpace> mdp;

    public ValueIteration(MDP<O, Integer, DiscreteSpace> mdp, ValueIteration.VIConfiguration conf) {
        this.mdp = mdp;
        this.conf = conf;
    }

    public Integer choose(O state) {
        if (RandomUtils.getRandom().nextDouble() <= this.conf.epsilon)
            return this.mdp.getActionSpace().randomAction();

        return this.mdp.getActionSpace().randomAction();
    }

    public void solve() {
        LOGGER.info("SOLVING");
    }

    public double play() {
        O state = this.mdp.reset();
        double rewards = 0;
        int i = 0;
        for (; !this.mdp.isDone(); i++) {
            StepReply<O> step = this.mdp.step(this.choose(state));
            state = step.getObservation();
            rewards += step.getReward();
        }
        return rewards / i;
    }

    public double evaluate(int episodes) {
        double rewards = 0;
        int i = 0;
        for (; i < episodes; i++) {
            double reward = this.play();
            rewards += reward;
            LOGGER.info("[Evaluate] Episode: " + i + "; Reward: " + reward);
        }

        LOGGER.info("[Evaluate] Average reward: " + rewards / i);
        return rewards / i;
    }

    @Data
    @AllArgsConstructor
    @Builder
    public static class VIConfiguration {
        int seed;
        int iterations;
        double epsilon;
        double gamma;
    }
}
