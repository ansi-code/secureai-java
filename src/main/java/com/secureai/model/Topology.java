package com.secureai.model;

import com.secureai.utils.IteratorUtils;
import lombok.Data;

import java.util.Map;

@Data
public class Topology {
    private String id;
    private Map<String, Node> nodes;
    private Map<String, Edge> edges;

    public long getInEdgesCount(String node) {
        return edges.values().stream().filter(edge -> edge.getTo().equals(node)).count();
    }

    public long getOutEdgesCount(String node) {
        return edges.values().stream().filter(edge -> edge.getFrom().equals(node)).count();
    }

    public Node getNode(int i) {
        return IteratorUtils.getInCollection(this.nodes.values(), i);
    }

    public String getNodeName(int i) {
        return IteratorUtils.getInSet(this.nodes.keySet(), i);
    }

    public void prettyPrint() {
        System.out.print("\033[H\033[2J");
        System.out.flush();
        int j = 0;
        for (String nodeName : nodes.keySet()) {
            for (int i = 0; i < nodes.get(nodeName).getReplication(); i++) {
                System.out.print(nodeName);
                System.out.print(String.format(" (%d)  ", j));
            }
            System.out.print("  ");
            j++;
        }
        System.out.print("\n");
    }
}
