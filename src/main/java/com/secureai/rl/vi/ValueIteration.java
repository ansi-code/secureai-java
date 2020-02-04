package com.secureai.rl.vi;

import com.secureai.rl.abs.DiscreteState;
import com.secureai.rl.abs.SMDP;
import lombok.*;
import org.deeplearning4j.gym.StepReply;
import org.deeplearning4j.rl4j.space.DiscreteSpace;

import java.util.logging.Logger;

public class ValueIteration<O extends DiscreteState> {

    private final static Logger LOGGER = Logger.getLogger(ValueIteration.class.getName());

    private ValueIteration.VIConfiguration conf;
    private SMDP<O, Integer, DiscreteSpace> mdp;

    private IntegerMap<Double> V; // State: Utility
    private IntegerMap<Integer> P; // State: Action

    @Getter
    @Setter
    private ValueIterationFilter<O> valueIterationFilter;

    public ValueIteration(SMDP<O, Integer, DiscreteSpace> mdp, ValueIteration.VIConfiguration conf) {
        //this(mdp, conf, new DynamicIntegerMap<>(), new DynamicIntegerMap<>());
        this(mdp, conf, new StaticIntegerMap<>((int) Math.pow(2, mdp.getObservationSpace().getShape()[0]), Double.class), new StaticIntegerMap<>((int) Math.pow(2, mdp.getObservationSpace().getShape()[0]), Integer.class));
    }

    public ValueIteration(SMDP<O, Integer, DiscreteSpace> mdp, ValueIteration.VIConfiguration conf, IntegerMap<Double> V, IntegerMap<Integer> P) {
        this.mdp = mdp;
        this.conf = conf;
        this.V = V;
        this.P = P;
    }

    public int chooseBest(O state) {
        int bestAction = -1;
        double bestQ = Double.NEGATIVE_INFINITY;
        double[] actionFilter = this.valueIterationFilter != null ? this.valueIterationFilter.run(state) : null;
        for (int a = 0; a < this.mdp.getActionSpace().getSize(); a++) {
            this.mdp.setState(state);
            StepReply<O> step = this.mdp.step(a);
            double q = step.getReward() + this.conf.gamma * this.V.getOrDefault(state.toInt(), 0d);
            q *= actionFilter != null ? actionFilter[a] : 1;
            if (q > bestQ) {
                bestQ = q;
                bestAction = a;
            }
        }

        return bestAction;
    }

    public int choose(O state) {
        return this.P.getOrDefault(state.toInt(), this.mdp.getActionSpace().randomAction());
    }

    public void solve() {
        int states = (int) Math.pow(2, this.mdp.getObservationSpace().getShape()[0]);
        LOGGER.info(String.format("[Solve] Starting iteration for %d states", states));
        for (int i = 0; i < this.conf.iterations; i++) {
            double vDelta = 0;
            for (int s = 0; s < states; s++) {
                this.mdp.getState().setFromInt(s);
                double previousV = this.V.getOrDefault(s, 0d);
                int bestAction = this.chooseBest(this.mdp.getState());
                StepReply<O> step = this.mdp.step(bestAction);
                this.V.put(s, step.getReward() + this.conf.gamma * this.V.getOrDefault(step.getObservation().toInt(), 0d));
                this.P.put(s, bestAction);
                vDelta = Math.max(vDelta, Math.abs(previousV - this.V.get(s)));
                if ((s + 1) % 10000 == 0 || (s + 1) == states) {
                    LOGGER.info(String.format("[Solve] State: %d/%d", (s + 1), states));
                    //this.play(); //uncomment if you want to see how it is going
                }
            }
            LOGGER.info(String.format("[Solve] Iteration: %d; Delta: %f", i, vDelta));

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
        LOGGER.info(String.format("[Play] Reward: %f; Average: %f", rewards, rewards / i));
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

    public interface ValueIterationFilter<O extends DiscreteState> {
        double[] run(O state);
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
