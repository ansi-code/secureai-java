package com.secureai.rl.agents;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class QNAgent {

    private INDArray qTable;

    public QNAgent(long stateSize, long actionSize) {
        this.qTable = Nd4j.zeros(stateSize, actionSize);
    }

    public learn() {
        //Q[state, action] = Q[state, action] + lr * (reward + gamma * np.max(Q[new_state, :]) — Q[state, action])
    }

    public int choose() {

        /*
        # Set the percent you want to explore
epsilon = 0.2
if random.uniform(0, 1) < epsilon:
    """
    Explore: select a random action
    """
else:
    """
    Exploit: select the action with max value (future reward)

         */

    }
}
