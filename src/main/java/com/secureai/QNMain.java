package com.secureai;

import com.secureai.model.topology.Topology;
import com.secureai.rl.qn.QLearning;
import com.secureai.system.SystemEnvironment;
import com.secureai.system.SystemState;
import com.secureai.utils.YAML;

import java.io.IOException;

public class QNMain {

    public static void main(String... args) throws IOException {
        Topology topology = YAML.parse("data/topology-2.yml", Topology.class);

        SystemEnvironment mdp = new SystemEnvironment(topology);

        QLearning.QNConfiguration qnConfiguration = new QLearning.QNConfiguration(
                123,    //Random seed
                2000,    //episodes
                64, //batch
                .628, //rate
                .9,     //discount
                .2    //espilon
        );

        QLearning<SystemState> ql = new QLearning<SystemState>(mdp, qnConfiguration);
        ql.train();
    }
}
