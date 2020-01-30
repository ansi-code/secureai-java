package com.secureai;

import com.secureai.model.actionset.ActionSet;
import com.secureai.model.topology.Topology;
import com.secureai.rl.vi.ValueIteration;
import com.secureai.system.SystemEnvironment;
import com.secureai.system.SystemState;
import com.secureai.utils.YAML;
import org.apache.log4j.BasicConfigurator;

import java.io.IOException;

public class VIMain {

    public static void main(String... args) throws IOException {
        BasicConfigurator.configure();

        Topology topology = YAML.parse("data/topologies/topology-1.yml", Topology.class);
        ActionSet actionSet = YAML.parse("data/action-sets/action-set-1.yml", ActionSet.class);

        SystemEnvironment mdp = new SystemEnvironment(topology, actionSet);

        ValueIteration.VIConfiguration viConfiguration = new ValueIteration.VIConfiguration(
                123,    //Random seed
                2000,    //iterations
                .5,    //gamma
                0.01    //epsilon
        );

        ValueIteration<SystemState> vi = new ValueIteration<>(mdp, viConfiguration);
        vi.solve();

        vi.evaluate(10);
    }
}
