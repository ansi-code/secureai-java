package com.secureai.model;

import lombok.Data;

import java.util.Map;

@Data
public class Topology {
    private String id;
    private Map<String, Node> nodes;
    private Map<String, Edge> edges;

    public void prettyPrint() {
        System.out.print("\033[H\033[2J");
        System.out.flush();
        for (String nodeName : nodes.keySet()) {
            System.out.print(nodeName);
            System.out.print("  ");
        }
        System.out.print("\n");
    }
}
