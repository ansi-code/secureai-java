package com.secureai.rl.qn;

import com.secureai.rl.abs.DiscreteState;
import com.secureai.utils.RandomUtils;
import com.secureai.utils.Stat;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import org.deeplearning4j.gym.StepReply;
import org.deeplearning4j.rl4j.mdp.MDP;
import org.deeplearning4j.rl4j.space.DiscreteSpace;

import java.util.logging.Logger;

public class QLearning<O extends DiscreteState> {

    private final static Logger LOGGER = Logger.getLogger(QLearning.class.getName());

    private QTable qTable;
    private QNConfiguration conf;
    private MDP<O, Integer, DiscreteSpace> mdp;

    public QLearning(MDP<O, Integer, DiscreteSpace> mdp, QNConfiguration conf) {
        //this.qTable = new StaticQTable((int) Math.pow(2, mdp.getObservationSpace().getShape()[0]), mdp.getActionSpace().getSize());
        this.qTable = new DynamicQTable(mdp.getActionSpace().getSize());
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

    public Integer choose(O state) {
        if (RandomUtils.getRandom().nextDouble() <= this.conf.epsilon)
            return this.mdp.getActionSpace().randomAction();

        return this.qTable.argMax(state.toInt());
    }

    public void train() {
        Stat<Double> stat = new Stat<>("output/qlearning.csv");

        for (int i = 0; i < this.conf.episodes; i++) { // episodes
            O state = this.mdp.reset();
            double rewards = 0;
            int j = 0;
            for (; j < this.conf.batchSize && !this.mdp.isDone(); j++) { // batches
                StepReply<O> step = this.trainStep(state);
                state = step.getObservation();

                rewards += step.getReward();
                stat.append(step.getReward());
            }
            LOGGER.info(String.format("[Train] Episode: %d; Reward: %f; Average: %f", i, rewards, rewards / j));
        }
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
        int batchSize;
        double learningRate;
        double discountFactor;
        double epsilon;
    }
}
