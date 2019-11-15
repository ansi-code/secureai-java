package com.secureai;

import com.secureai.nn.NNBuilder;
import com.secureai.rl.rl4j.SystemEnv;
import com.secureai.rl.rl4j.SystemState;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.rl4j.learning.sync.qlearning.QLearning;
import org.deeplearning4j.rl4j.learning.sync.qlearning.discrete.QLearningDiscreteDense;
import org.deeplearning4j.rl4j.network.dqn.DQN;
import org.deeplearning4j.rl4j.policy.DQNPolicy;
import org.deeplearning4j.rl4j.util.DataManager;

import java.io.IOException;

public class Main {

    public static void main(String... args) throws IOException {
        //System.out.println("Hello World");
        //System.out.println(YAML.parse("data/topology-1.yml", Topology.class));
        //YAML.parse("data/topology-1.yml", Topology.class).prettyPrint();

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
                true    //double DQN
        );

        DataManager manager = new DataManager(true);

        SystemEnv mdp = new SystemEnv();
        MultiLayerNetwork nn = new NNBuilder().build(10, 1);

        QLearningDiscreteDense<SystemState> dql = new QLearningDiscreteDense<>(mdp, new DQN(nn), qlConfiguration, manager);
        dql.train();

        DQNPolicy<SystemState> pol = dql.getPolicy();
        pol.save("/tmp/pol1");
        mdp.close();
    }
}
