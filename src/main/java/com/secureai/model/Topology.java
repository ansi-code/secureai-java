package com.secureai.model;

import java.util.Map;

public class Topology {
    private String id;
    private Map<String, Node> nodes;

    public String getId() {
        return id;
    }

    public void setId(String id) {
        this.id = id;
    }

    public Map<String, Node> getNodes() {
        return nodes;
    }

    public void setNodes(Map<String, Node> nodes) {
        this.nodes = nodes;
    }

    @Override
    public String toString() {
        return "Topology{" +
                "id='" + id + '\'' +
                ", nodes=" + nodes +
                '}';
    }
}
