package com.secureai.rl.vi;

import com.secureai.rl.abs.DiscreteState;
import com.secureai.utils.RandomUtils;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import org.deeplearning4j.gym.StepReply;
import org.deeplearning4j.rl4j.mdp.MDP;
import org.deeplearning4j.rl4j.space.DiscreteSpace;

import java.util.HashMap;
import java.util.logging.Logger;

public class ValueIteration<O extends DiscreteState> {

    private final static Logger LOGGER = Logger.getLogger(ValueIteration.class.getName());

    private ValueIteration.VIConfiguration conf;
    private MDP<O, Integer, DiscreteSpace> mdp;

    private HashMap<Integer, Double> V = new HashMap<>(); // State: Utility
    private HashMap<Integer, Integer> P = new HashMap<>(); // State: Action

    public ValueIteration(MDP<O, Integer, DiscreteSpace> mdp, ValueIteration.VIConfiguration conf) {
        this.mdp = mdp;
        this.conf = conf;
    }

    public int choose(O state) {
        int bestAction = -1;
        double bestQ = -Double.MAX_VALUE;
        for (int a = 0; a < this.mdp.getActionSpace().getSize(); a++) {
            //this.mdp.setState(state);
            StepReply<O> step = this.mdp.step(a);
            double q = step.getReward() + this.conf.gamma * this.V.get(state.toInt());
            if (q > bestQ) {
                bestQ = q;
                bestAction = a;
            }
        }

        //bestAction = P.get(state); // In test mode

        return bestAction;
    }

    public void solve() {
        LOGGER.info("SOLVING");
        for (int i = 0; i < this.conf.iterations; i++) {
            double vDelta = 0;

            for (int s = 0; s < this.mdp.getObservationSpace().getShape()[0]; s++) {
                double previousV = V.get(s);
                //int bestAction = this.choose(s);
                //this.mdp.setState(j);
                //StepReply<O> step = this.mdp.step(bestAction);
                //this.V.put(s, step.getReward() + this.conf.gamma * this.V.get(step.getObservation()));
                //this.P.put(s, bestAction);
                vDelta = Math.max(vDelta, Math.abs(previousV - this.V.get(s)));
            }

            if (vDelta < this.conf.epsilon)
                break;
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
        double gamma;
        double epsilon;
    }
}
