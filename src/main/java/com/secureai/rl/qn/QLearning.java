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
            for (int j = 0; j < this.conf.batchSize && !this.mdp.isDone(); j++) { // batches
                StepReply<O> step = this.trainStep(state);
                state = step.getObservation();

                LOGGER.info("Reward: " + step.getReward());
                stat.append(step.getReward());
            }
        }
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
