package com.secureai;

import com.secureai.model.actionset.ActionSet;
import com.secureai.model.topology.Topology;
import com.secureai.nn.DynNNBuilder;
import com.secureai.nn.NNBuilder;
import com.secureai.system.SystemEnvironment;
import com.secureai.system.SystemState;
import com.secureai.utils.*;
import lombok.SneakyThrows;
import org.apache.log4j.BasicConfigurator;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.rl4j.learning.IEpochTrainer;
import org.deeplearning4j.rl4j.learning.sync.qlearning.QLearning;
import org.deeplearning4j.rl4j.learning.sync.qlearning.discrete.QLearningDiscreteDense;
import org.deeplearning4j.rl4j.network.dqn.DQN;
import org.deeplearning4j.rl4j.util.DataManager;
import org.deeplearning4j.rl4j.util.DataManagerTrainingListener;
import org.deeplearning4j.rl4j.util.IDataManager.StatEntry;

import java.io.IOException;
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
        BasicConfigurator.configure();

        runWithThreshold();
        //runWithTimer();

        for (; ; ) queue.take().run();
    }

    public static void runWithThreshold() {
        int EPOCH_THRESHOLD = 5; // After 5 epochs

        DynDQNMain.setup();

        dql.addListener(new EpochEndListener() {
            @Override
            public ListenerResponse onEpochTrainingResult(IEpochTrainer iEpochTrainer, StatEntry statEntry) {
                if (iEpochTrainer.getEpochCounter() == EPOCH_THRESHOLD) {
                    System.out.println("THRESHOLD FIRED");
                    Timer t = new Timer();
                    t.schedule(new TimerTask() {
                        @Override
                        public void run() {
                            DynDQNMain.stop(DynDQNMain::runWithThreshold);
                            t.cancel();
                        }
                    }, 5000);
                }
                return null;
            }
        });

        queue.add(dql::train);
    }

    public static void runWithTimer() {
        int TIMER_THRESHOLD = 15000; // After 0s and period 15s

        new Timer(true).schedule(new TimerTask() {
            @SneakyThrows
            @Override
            public void run() {
                System.out.println("TIMER FIRED");
                DynDQNMain.stop(() -> {
                    DynDQNMain.setup();
                    queue.add(dql::train);
                });
            }
        }, 0, TIMER_THRESHOLD);
    }

    public static void stop(CallbackUtils.NoArgsCallback callback) {
        if (dql != null) {
            dql.addListener(new TrainingEndListener() {
                @Override
                public void onTrainingEnd() {
                    callback.callback();
                }
            });
            dql.getConfiguration().setMaxStep(0);
            dql.getConfiguration().setMaxEpochStep(0);
        } else {
            callback.callback();
        }
    }

    public static void setup() {
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
        nn.setListeners(new ScoreIterationListener(100));
        mdp = newMdp;

        dql = new QLearningDiscreteDense<>(mdp, new DQN<>(nn), qlConfiguration);
        //dql = new QLearningDiscreteDense<>(mdp, new ParallelDQN<>(nn), qlConfiguration);
        //dql = new QLearningDiscreteDense<>(mdp, new SparkDQN<>(nn), qlConfiguration);
        try {
            DataManager dataManager = new DataManager(true);
            dql.addListener(new DataManagerTrainingListener(dataManager));
            dql.addListener(new RLStatTrainingListener(dataManager.getInfo().substring(0, dataManager.getInfo().lastIndexOf('/'))));
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
