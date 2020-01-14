package com.secureai;

import com.secureai.model.actionset.ActionSet;
import com.secureai.model.topology.Topology;
import com.secureai.utils.YAML;

public class DataTest {

    public static void main(String[] args) throws Exception {
        Topology topology = YAML.parse("data/topologies/topology-2.yml", Topology.class);
        System.out.println(topology);

        ActionSet actionSet = YAML.parse("data/action-sets/action-set-1.yml", ActionSet.class);
        System.out.println(actionSet);
    }
}
