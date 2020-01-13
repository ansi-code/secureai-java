package com.secureai.system;

import com.secureai.model.topology.Resource;
import com.secureai.model.topology.Topology;
import com.secureai.utils.IteratorUtils;
import lombok.Getter;

import java.util.HashMap;
import java.util.Map;

public class SystemDefinition {
    @Getter
    private Topology topology;

    public SystemDefinition(Topology topology) {
        this.topology = topology;
    }

    public Map<String, Integer> getMap() {
        HashMap<String, Integer> map = new HashMap<>();
        int i = 0;
        for (String key : this.topology.getResources().keySet()) {
            map.put(key, i);
            i++;
        }
        return map;
    }

    public long getInConnectionsCount(String node) {
        return this.topology.getConnections().values().stream().filter(edge -> edge.getTo().equals(node)).count();
    }

    public long getOutConnectionsCount(String node) {
        return this.topology.getConnections().values().stream().filter(edge -> edge.getFrom().equals(node)).count();
    }

    public Resource getNode(int i) {
        return IteratorUtils.getInCollection(this.topology.getResources().values(), i);
    }

    public String getNodeName(int i) {
        return IteratorUtils.getInSet(this.topology.getResources().keySet(), i);
    }

    public void prettyPrint() {
        System.out.print("\033[H\033[2J");
        System.out.flush();
        int j = 0;
        for (String nodeName : this.topology.getResources().keySet()) {
            for (int i = 0; i < this.topology.getResources().get(nodeName).getReplication(); i++) {
                System.out.print(nodeName);
                System.out.print(String.format(" (%d)  ", j));
            }
            System.out.print("  ");
            j++;
        }
        System.out.print("\n");
    }
}
