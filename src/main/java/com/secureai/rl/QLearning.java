package com.secureai.rl;

import com.secureai.rl.abs.DiscreteState;
import com.secureai.utils.RandomUtils;
import com.secureai.utils.Stat;
import org.deeplearning4j.gym.StepReply;
import org.deeplearning4j.rl4j.mdp.MDP;
import org.deeplearning4j.rl4j.space.DiscreteSpace;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class QLearning<O extends DiscreteState> {

    private INDArray qTable;
    private double learningRate;
    private double discountFactor;
    private MDP<O, Integer, DiscreteSpace> mdp;


    public QLearning(MDP<O, Integer, DiscreteSpace> mdp, double learningRate, double discountFactor) {
        this.qTable = Nd4j.rand((int) Math.pow(2, mdp.getObservationSpace().getShape()[0]), mdp.getActionSpace().getSize());
        this.mdp = mdp;
        this.learningRate = learningRate;
        this.discountFactor = discountFactor;
    }

    public StepReply<O> trainStep(O state) {
        Integer action = this.choose(state);
        StepReply<O> step = this.mdp.step(action);

        int oldState = state.toInt();
        int newState = step.getObservation().toInt();
        double reward = step.getReward();

        this.qTable.put(oldState, action, this.qTable.getDouble(oldState, action) + this.learningRate * (reward + this.discountFactor * this.qTable.getRow(newState).maxNumber().doubleValue() - this.qTable.getDouble(oldState, action)));

        return step;
    }

    public Integer choose(O state) {
        double epsilon = 0.2;
        if (RandomUtils.getRandom().nextDouble() <= epsilon)
            return this.mdp.getActionSpace().randomAction();

        return this.qTable.getRow(state.toInt()).argMax().getInt(0);
    }

    public void train() {
        Stat<Double> stat = new Stat<>("output/qlearning_1.csv");
        for (int i = 0; i < 99; i++) { // episodes
            O state = this.mdp.reset();
            for (int j = 0; j < 99 && !this.mdp.isDone(); j++) { // batches
                StepReply<O> step = this.trainStep(state);
                state = step.getObservation();
                System.out.println("Reward: " + String.valueOf(step.getReward()));
                stat.append(step.getReward());
            }
        }
    }
}
