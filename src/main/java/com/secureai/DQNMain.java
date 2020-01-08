package com.secureai;

import com.secureai.model.Topology;
import com.secureai.nn.NNBuilder;
import com.secureai.system.SystemEnvironment;
import com.secureai.system.SystemState;
import com.secureai.utils.YAML;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.rl4j.learning.sync.qlearning.QLearning;
import org.deeplearning4j.rl4j.learning.sync.qlearning.discrete.QLearningDiscreteDense;
import org.deeplearning4j.rl4j.network.dqn.DQN;
import org.deeplearning4j.rl4j.policy.DQNPolicy;
import org.deeplearning4j.rl4j.util.DataManager;

import java.io.IOException;

public class DQNMain {

    public static void main(String... args) throws IOException {
        //System.out.println("Hello World");
        //System.out.println(YAML.parse("data/topology-1.yml", Topology.class));
        //YAML.parse("data/topology-1.yml", Topology.class).prettyPrint();

        Topology topology = YAML.parse("data/topology-1.yml", Topology.class);

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

        SystemEnvironment mdp = new SystemEnvironment(topology);
        MultiLayerNetwork nn = new NNBuilder().build(mdp.getObservationSpace().size(), mdp.getActionSpace().size());

        QLearningDiscreteDense<SystemState> dql = new QLearningDiscreteDense<>(mdp, new DQN<>(nn), qlConfiguration, manager);
        dql.train();

        DQNPolicy<SystemState> pol = dql.getPolicy();
        pol.save("/tmp/pol1");
        mdp.close();
    }
}
