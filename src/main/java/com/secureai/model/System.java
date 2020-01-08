package com.secureai.model;

import java.util.HashMap;
import java.util.Map;
import java.util.stream.Collectors;

public class System {
    private Topology topology;

    public System(Topology topology) {
        this.topology = topology;
    }

    public Map<String, Integer> getMap() {
        HashMap<String, Integer> map = new HashMap<>();
        int i = 0;
        for (String key: this.topology.getNodes().keySet()) {
            map.put(key, i);
            i++;
        }
        return map;
    }
}
