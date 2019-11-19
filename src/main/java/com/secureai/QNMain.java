package com.secureai;

import com.secureai.model.Topology;
import com.secureai.rl.QLearning;
import com.secureai.rl.rl4j.SystemEnvironment;
import com.secureai.rl.rl4j.SystemState;
import com.secureai.utils.YAML;

import java.io.IOException;

public class QNMain {

    public static void main(String... args) throws IOException {
        Topology topology = YAML.parse("data/topology-2.yml", Topology.class);

        SystemEnvironment mdp = new SystemEnvironment(topology);

        QLearning<SystemState> ql = new QLearning(mdp, .628, .9);
        ql.train();
    }
}
