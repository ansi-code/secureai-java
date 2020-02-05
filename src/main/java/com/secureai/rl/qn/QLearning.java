package com.secureai.rl.qn;

import com.secureai.rl.abs.DiscreteState;
import com.secureai.utils.RandomUtils;
import com.secureai.utils.Stat;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.Getter;
import org.deeplearning4j.gym.StepReply;
import org.deeplearning4j.rl4j.mdp.MDP;
import org.deeplearning4j.rl4j.space.DiscreteSpace;

import java.util.logging.Logger;

public class QLearning<O extends DiscreteState> {

    private final static Logger LOGGER = Logger.getLogger(QLearning.class.getName());

    private QTable qTable;
    private QNConfiguration conf;
    private MDP<O, Integer, DiscreteSpace> mdp;

    @Getter
    private int stepCounter = 0;

    public QLearning(MDP<O, Integer, DiscreteSpace> mdp, QNConfiguration conf) {
        this(mdp, conf, new DynamicQTable(mdp.getActionSpace().getSize()));
    }

    public QLearning(MDP<O, Integer, DiscreteSpace> mdp, QNConfiguration conf, QTable qTable) {
        this.qTable = qTable;
        this.mdp = mdp;
        this.conf = conf;
    }

    public StepReply<O> trainStep(O state) {
        Integer action = this.choose(state);
        StepReply<O> step = this.mdp.step(action);

        int oldState = state.toInt();
        int newState = step.getObservation().toInt();
        double reward = step.getReward();
        this.qTable.set(oldState, action, this.qTable.get(oldState, action) + this.conf.learningRate * (reward + this.conf.discountFactor * this.qTable.max(newState) - this.qTable.get(oldState, action)));

        return step;
    }

    public Integer choosePolicy(O state) {
        return Math.max(this.qTable.argMax(state.toInt()), 0);
    }

    public Integer choose(O state) {
        return RandomUtils.getRandom().nextFloat() > this.getEpsilon() ? this.choosePolicy(state) : this.mdp.getActionSpace().randomAction();
    }

    public float getEpsilon() {
        return Math.min(1.0F, Math.max(this.conf.minEpsilon, 1.0F - (float) (this.getStepCounter()/* - this.updateStart*/) * 1.0F / (float) this.conf.epsilonNbStep));
    }

    public void train() {
        Stat<Double> stat = new Stat<>("output/q_learning.csv");

        for (int i = 0; i < this.conf.episodes; i++) { // episodes
            O state = this.mdp.reset();
            double rewards = 0;
            int j = 0;
            for (; j < this.conf.maxEpochStep && !this.mdp.isDone(); j++) { // batches
                StepReply<O> step = this.trainStep(state);
                state = step.getObservation();

                rewards += step.getReward();
                this.stepCounter++;
            }
            stat.append(rewards);
            LOGGER.info(String.format("[Train] Episode: %d; Reward: %f; Average: %f", i, rewards, rewards / j));
        }
        stat.flush();
    }

    public double play() {
        O state = this.mdp.reset();
        double rewards = 0;
        int i = 0;
        for (; !this.mdp.isDone(); i++) {
            StepReply<O> step = this.mdp.step(this.choosePolicy(state));
            state = step.getObservation();
            rewards += step.getReward();
        }
        LOGGER.info(String.format("[Play] Episode: %d; Reward: %f; Average: %f", i, rewards, rewards / i));
        return rewards;
    }

    public double evaluate(int episodes) {
        double rewards = 0;
        int i = 0;
        for (; i < episodes; i++) {
            double reward = this.play();
            rewards += reward;
        }
        LOGGER.info(String.format("[Evaluate] Average: %f", rewards / i));
        return rewards / i;
    }

    @Data
    @AllArgsConstructor
    @Builder
    public static class QNConfiguration {
        int seed;
        int episodes;
        int maxEpochStep;
        double learningRate;
        double discountFactor;
        float minEpsilon;
        int epsilonNbStep;
    }
}
