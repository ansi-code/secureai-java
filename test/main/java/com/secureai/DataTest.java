package com.secureai;

import com.secureai.model.actionset.ActionSet;
import com.secureai.model.stateset.State;
import com.secureai.model.topology.Topology;
import com.secureai.system.SystemEnvironment;
import com.secureai.system.SystemState;
import com.secureai.utils.ArrayUtils;
import com.secureai.utils.YAML;

import java.util.Arrays;

public class DataTest {

    public static void main(String[] args) throws Exception {
        Topology topology = YAML.parse("data/topologies/topology-paper-4.yml", Topology.class);
        System.out.println(topology);

        ActionSet actionSet = YAML.parse("data/action-sets/action-set-paper.yml", ActionSet.class);
        System.out.println(actionSet);

        System.out.println(ArrayUtils.toBase10(new int[]{0, 1, 0, 1}, 2));
        System.out.println(Arrays.toString(ArrayUtils.fromBase10(135, 2)));

        SystemState systemState = new SystemState(new SystemEnvironment(topology, actionSet));
        systemState.reset();
        SystemState newSystemState = systemState.newInstance(systemState.toInt());
        System.out.println(Arrays.toString(systemState.toArray()));
        System.out.println(systemState.equals(newSystemState));
        systemState.set("api-gateway-main.0", State.active, true);
        System.out.println(systemState.equals(newSystemState));
        systemState.setFromInt(newSystemState.toInt());
        System.out.println(systemState.equals(newSystemState));;
    }
}
