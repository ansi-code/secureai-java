package com.secureai;

import com.secureai.model.actionset.ActionSet;
import com.secureai.model.topology.Topology;
import com.secureai.nn.FilteredMultiLayerNetwork;
import com.secureai.nn.NNBuilder;
import com.secureai.system.SystemEnvironment;
import com.secureai.system.SystemState;
import com.secureai.utils.RLStatTrainingListener;
import com.secureai.utils.YAML;
import org.apache.log4j.BasicConfigurator;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.rl4j.learning.sync.qlearning.QLearning;
import org.deeplearning4j.rl4j.learning.sync.qlearning.discrete.QLearningDiscreteDense;
import org.deeplearning4j.rl4j.network.dqn.DQN;
import org.deeplearning4j.rl4j.util.DataManager;
import org.deeplearning4j.rl4j.util.DataManagerTrainingListener;

import java.io.IOException;
import java.util.logging.Logger;

public class DQNMain {

    public static void main(String... args) throws IOException {
        BasicConfigurator.configure();

        Topology topology = YAML.parse("data/topologies/topology-1.yml", Topology.class);
        ActionSet actionSet = YAML.parse("data/action-sets/action-set-test.yml", ActionSet.class);

        QLearning.QLConfiguration qlConfiguration = new QLearning.QLConfiguration(
                123,    //Random seed
                50,    //Max step By epoch
                1000, //Max step
                150000, //Max size of experience replay
                32,     //size of batches
                500,    //target update (hard)
                10,     //num step noop warmup
                1.,   //reward scaling
                0.99,   //gamma
                1.0,    //td-error clipping
                0.1f,   //min epsilon
                1000,   //num step for eps greedy anneal
                false    //double DQN
        );

        SystemEnvironment mdp = new SystemEnvironment(topology, actionSet);
        FilteredMultiLayerNetwork nn = new NNBuilder().build(mdp.getObservationSpace().size(), mdp.getActionSpace().getSize());

        nn.setMultiLayerNetworkPredictionFilter(input -> mdp.getActionSpace().actionsMask(input));
        System.out.println(nn.summary());
        nn.setListeners(new ScoreIterationListener(100));

        QLearningDiscreteDense<SystemState> dql = new QLearningDiscreteDense<>(mdp, new DQN<>(nn), qlConfiguration);
        //QLearningDiscreteDense<SystemState> dql = new QLearningDiscreteDense<>(mdp, new ParallelDQN<>(nn), qlConfiguration);
        //QLearningDiscreteDense<SystemState> dql = new QLearningDiscreteDense<>(mdp, new SparkDQN<>(nn), qlConfiguration);
        DataManager dataManager = new DataManager(true);
        dql.addListener(new DataManagerTrainingListener(dataManager));
        dql.addListener(new RLStatTrainingListener(dataManager.getInfo().substring(0, dataManager.getInfo().lastIndexOf('/'))));
        dql.train();

        int EPISODES = 10;
        double rewards = 0;
        for (int i = 0; i < EPISODES; i++) {
            mdp.reset();
            double reward = dql.getPolicy().play(mdp);
            rewards += reward;
            Logger.getAnonymousLogger().info("[Evaluate] Reward: " + reward);
        }
        Logger.getAnonymousLogger().info("[Evaluate] Average reward: " + rewards / EPISODES);
    }
}
