package com.secureai;

import com.secureai.model.actionset.ActionSet;
import com.secureai.model.topology.Topology;
import com.secureai.nn.DynNNBuilder;
import com.secureai.nn.NNBuilder;
import com.secureai.system.SystemEnvironment;
import com.secureai.system.SystemState;
import com.secureai.utils.RandomUtils;
import com.secureai.utils.YAML;
import lombok.SneakyThrows;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.rl4j.learning.sync.qlearning.QLearning;
import org.deeplearning4j.rl4j.learning.sync.qlearning.discrete.QLearningDiscreteDense;
import org.deeplearning4j.rl4j.network.dqn.DQN;

import java.util.Timer;
import java.util.TimerTask;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.LinkedBlockingQueue;

public class DynDQNMain {

    public static final BlockingQueue<Runnable> queue = new LinkedBlockingQueue<>();
    static QLearningDiscreteDense<SystemState> dql = null;
    static MultiLayerNetwork nn = null;
    static SystemEnvironment mdp = null;

    public static void main(String... args) throws InterruptedException {

        Timer timer = new Timer(true);
        timer.schedule(new TimerTask() {
            @SneakyThrows
            @Override
            public void run() {
                System.out.println("TIMER FIRED");
                if (dql != null) {
                    dql.getConfiguration().setMaxStep(0);
                    dql = null;
                    queue.clear();
                }

                Topology topology = YAML.parse(String.format("data/topologies/topology-%d.yml", RandomUtils.getRandom().nextDouble() >= .5 ? 1 : 2), Topology.class);
                ActionSet actionSet = YAML.parse(String.format("data/action-sets/action-set-%d.yml", RandomUtils.getRandom().nextDouble() >= .5 ? 1 : 2), ActionSet.class);

                QLearning.QLConfiguration qlConfiguration = new QLearning.QLConfiguration(
                        123,    //Random seed
                        200,    //Max step By epoch
                        150000, //Max step
                        150000, //Max size of experience replay
                        32,     //size of batches
                        500,    //target update (hard)
                        10,     //num step noop warmup
                        0.01,   //reward scaling
                        0.99,   //gamma
                        1.0,    //td-error clipping
                        0.1f,   //min epsilon
                        1000,   //num step for eps greedy anneal
                        false    //double DQN
                );

                SystemEnvironment newMdp = new SystemEnvironment(topology, actionSet);
                if (nn == null)
                    nn = new NNBuilder().build(newMdp.getObservationSpace().size(), newMdp.getActionSpace().getSize());
                else
                    nn = new DynNNBuilder(nn)
                            .forLayer(0).transferIn(mdp.getObservationSpace().getMap(), newMdp.getObservationSpace().getMap())
                            .forLayer(-1).transferOut(mdp.getActionSpace().getMap(), newMdp.getActionSpace().getMap())
                            .build();
                // System.out.println(nn.summary());
                mdp = newMdp;

                queue.add(() -> {
                    dql = new QLearningDiscreteDense<>(mdp, new DQN<>(nn.clone()), qlConfiguration);
                    dql.train();
                });

            }
        }, 0, 15000); // After 0s and period 10s

        for (; ; ) queue.take().run();
    }
}
