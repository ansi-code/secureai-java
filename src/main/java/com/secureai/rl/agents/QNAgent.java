package com.secureai.rl.agents;

import com.secureai.rl.abs.Action;
import com.secureai.rl.abs.ActionSpace;
import com.secureai.rl.abs.State;
import com.secureai.rl.abs.StateSpace;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Random;

public class QNAgent {

    private INDArray qTable;
    private StateSpace stateSpace;
    private ActionSpace actionSpace;
    private double learningRate;
    private double discountFactor;

    private Random random = new Random(12345);

    public QNAgent(StateSpace stateSpace, ActionSpace actionSpace, double learningRate, double discountFactor) {
        this.qTable = Nd4j.rand(stateSpace.getSize(), actionSpace.getSize());
        this.stateSpace = stateSpace;
        this.actionSpace = actionSpace;
        this.learningRate = learningRate;
        this.discountFactor = discountFactor;
    }

    public void train(State state, Action action, State newState, double reward) {
        int stateIndex = this.stateSpace.indexOf(state);
        int actionIndex = this.actionSpace.indexOf(action);
        int newStateIndex = this.stateSpace.indexOf(newState);

        this.qTable.put(stateIndex, actionIndex, this.qTable.getDouble(stateIndex, actionIndex) + this.learningRate * (reward + this.discountFactor * this.qTable.getColumn(newStateIndex).maxNumber().doubleValue() - this.qTable.getDouble(stateIndex, actionIndex)));
    }

    public Action choose(State state) {
        double epsilon = 0.2;
        if (this.random.nextDouble() <= epsilon) {
            int randomActionIndex = random.nextInt(actionSpace.getSize());
            return this.actionSpace.get(randomActionIndex);
        }

        int stateIndex = this.stateSpace.indexOf(state);
        int maxActionIndex = this.qTable.getColumn(stateIndex).argMax().getInt(0);
        return this.actionSpace.get(maxActionIndex);
    }
}
