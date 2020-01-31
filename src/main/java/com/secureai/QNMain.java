package com.secureai;

import com.secureai.model.actionset.ActionSet;
import com.secureai.model.topology.Topology;
import com.secureai.rl.qn.FilteredDynamicQTable;
import com.secureai.rl.qn.QLearning;
import com.secureai.system.SystemEnvironment;
import com.secureai.system.SystemState;
import com.secureai.utils.YAML;
import org.apache.commons.lang3.ArrayUtils;
import org.apache.log4j.BasicConfigurator;

import java.io.IOException;

public class QNMain {

    public static void main(String... args) throws IOException {
        BasicConfigurator.configure();

        Topology topology = YAML.parse("data/topologies/topology-1.yml", Topology.class);
        ActionSet actionSet = YAML.parse("data/action-sets/action-set-test.yml", ActionSet.class);

        SystemEnvironment mdp = new SystemEnvironment(topology, actionSet);

        QLearning.QNConfiguration qnConfiguration = new QLearning.QNConfiguration(
                123,    //Random seed
                2000,    //episodes
                64, //batch
                .628, //rate
                .9,     //discount
                .2    //espilon
        );

        FilteredDynamicQTable qTable = new FilteredDynamicQTable(mdp.getActionSpace().getSize());
        qTable.setDynamicQTableGetFilter(input -> ArrayUtils.toPrimitive(mdp.getActionSpace().actionsMask(input)));

        QLearning<SystemState> ql = new QLearning<>(mdp, qnConfiguration);
        ql.train();

        ql.evaluate(10);
    }
}
