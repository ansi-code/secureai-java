package com.secureai;

import com.secureai.model.actionset.ActionSet;
import com.secureai.model.topology.Topology;
import com.secureai.rl.qn.FilteredDynamicQTable;
import com.secureai.rl.qn.QLearning;
import com.secureai.system.SystemEnvironment;
import com.secureai.system.SystemState;
import com.secureai.utils.ArgsUtils;
import com.secureai.utils.YAML;
import org.apache.commons.lang3.ArrayUtils;
import org.apache.log4j.BasicConfigurator;

import java.io.IOException;
import java.util.Map;

public class QNMain {

    public static void main(String... args) throws IOException {
        BasicConfigurator.configure();
        Map<String, String> argsMap = ArgsUtils.toMap(args);

        Topology topology = YAML.parse(String.format("data/topologies/topology-%s.yml", argsMap.getOrDefault("topology", "paper-4")), Topology.class);
        ActionSet actionSet = YAML.parse(String.format("data/action-sets/action-set-%s.yml", argsMap.getOrDefault("actionSet", "paper")), ActionSet.class);

        SystemEnvironment mdp = new SystemEnvironment(topology, actionSet);

        QLearning.QNConfiguration qnConfiguration = new QLearning.QNConfiguration(
                Integer.parseInt(argsMap.getOrDefault("seed", "123")),          //Random seed
                Integer.parseInt(argsMap.getOrDefault("episodes", "100000")),     //episodes
                Integer.parseInt(argsMap.getOrDefault("batchSize", "64")),      //batch
                Double.parseDouble(argsMap.getOrDefault("learningRate", ".628")), //rate
                Double.parseDouble(argsMap.getOrDefault("discountFactor", "1")), //discount
                Double.parseDouble(argsMap.getOrDefault("epsilon", ".1"))        //espilon
        );

        FilteredDynamicQTable qTable = new FilteredDynamicQTable(mdp.getActionSpace().getSize());
        qTable.setDynamicQTableGetFilter(input -> ArrayUtils.toPrimitive(mdp.getActionSpace().actionsMask(input)));

        QLearning<SystemState> ql = new QLearning<>(mdp, qnConfiguration);
        ql.train();

        ql.evaluate(10);
    }
}
