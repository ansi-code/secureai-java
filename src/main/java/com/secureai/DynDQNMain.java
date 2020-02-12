package com.secureai;

import com.secureai.model.actionset.ActionSet;
import com.secureai.model.topology.Topology;
import com.secureai.nn.DynNNBuilder;
import com.secureai.nn.NNBuilder;
import com.secureai.rl.abs.ParallelDQN;
import com.secureai.rl.abs.SparkDQN;
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
import java.util.Map;
import java.util.Timer;
import java.util.TimerTask;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.LinkedBlockingQueue;

public class DynDQNMain {

    public static final BlockingQueue<Runnable> queue = new LinkedBlockingQueue<>();
    static QLearningDiscreteDense<SystemState> dql = null;
    static MultiLayerNetwork nn = null;
    static SystemEnvironment mdp = null;
    static Map<String, String> argsMap;
    static int switches = 0;

    public static void main(String... args) throws InterruptedException {
        System.setProperty("org.bytedeco.javacpp.maxphysicalbytes", "0");
        System.setProperty("org.bytedeco.javacpp.maxbytes", "0");
        BasicConfigurator.configure();
        TimeUtils.setupStartMillis();

        argsMap = ArgsUtils.toMap(args);

        runWithThreshold();
        //runWithTimer();

        for (; ; ) queue.take().run();
    }

    public static void runWithThreshold() {
        int EPOCH_THRESHOLD = 500; // After 500 epochs

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
        if (switches++ > 2) System.exit(0);
        String topologyId = switches == 1 ? "0" : "0"; // RandomUtils.getRandom(new String[]{"paper-4", "paper-7"});
        String actionSetId = switches == 1 ? "paper-5" : "paper-6"; // RandomUtils.getRandom(new String[]{"paper-5", "paper-6"});
        System.out.println(String.format("[Dyn] Choosing topology '%s' with action set '%s'", topologyId, actionSetId));

        Topology topology = YAML.parse(String.format("data/topologies/topology-%s.yml", topologyId), Topology.class);
        ActionSet actionSet = YAML.parse(String.format("data/action-sets/action-set-%s.yml", actionSetId), ActionSet.class);

        QLearning.QLConfiguration qlConfiguration = new QLearning.QLConfiguration(
                Integer.parseInt(argsMap.getOrDefault("seed", "123")),                //Random seed
                Integer.parseInt(argsMap.getOrDefault("maxEpochStep", "1000")),       //Max step By epoch
                Integer.parseInt(argsMap.getOrDefault("maxStep", "20000")),           //Max step
                Integer.parseInt(argsMap.getOrDefault("expRepMaxSize", "5000")),      //Max size of experience replay
                Integer.parseInt(argsMap.getOrDefault("batchSize", "128")),           //size of batches
                Integer.parseInt(argsMap.getOrDefault("targetDqnUpdateFreq", "500")), //target update (hard)
                Integer.parseInt(argsMap.getOrDefault("updateStart", "100")),         //num step noop warmup
                Double.parseDouble(argsMap.getOrDefault("rewardFactor", "1")),        //reward scaling
                Double.parseDouble(argsMap.getOrDefault("gamma", "0.25")),            //gamma
                Double.parseDouble(argsMap.getOrDefault("errorClamp", "0.5")),        //td-error clipping
                Float.parseFloat(argsMap.getOrDefault("minEpsilon", "0.01")),         //min epsilon
                Integer.parseInt(argsMap.getOrDefault("epsilonNbStep", "1000")),     //num step for eps greedy anneal
                Boolean.parseBoolean(argsMap.getOrDefault("doubleDQN", "false"))      //double DQN
        );

        SystemEnvironment newMdp = new SystemEnvironment(topology, actionSet);
        if (nn == null)
            nn = new NNBuilder().build(newMdp.getObservationSpace().size(), newMdp.getActionSpace().getSize(), Integer.parseInt(argsMap.getOrDefault("layers", "3")));
        else
            nn = new DynNNBuilder<>((MultiLayerNetwork) dql.getNeuralNet().getNeuralNetworks()[0])
                    .forLayer(0).transferIn(mdp.getObservationSpace().getMap(), newMdp.getObservationSpace().getMap())
                    .forLayer(-1).transferOut(mdp.getActionSpace().getMap(), newMdp.getActionSpace().getMap())
                    .build();

        //nn.setMultiLayerNetworkPredictionFilter(input -> mdp.getActionSpace().actionsMask(input));
        nn.setListeners(new ScoreIterationListener(100));
        //nn.setListeners(new PerformanceListener(1, true, true));
        System.out.println(nn.summary());

        mdp = newMdp;

        String dqnType = argsMap.getOrDefault("dqn", "standard");
        dql = new QLearningDiscreteDense<>(mdp, dqnType.equals("parallel") ? new ParallelDQN<>(nn) : dqnType.equals("spark") ? new SparkDQN<>(nn) : new DQN<>(nn), qlConfiguration);
        try {
            DataManager dataManager = new DataManager(true);
            dql.addListener(new DataManagerTrainingListener(dataManager));
            dql.addListener(new RLStatTrainingListener(dataManager.getInfo().substring(0, dataManager.getInfo().lastIndexOf('/'))));
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
