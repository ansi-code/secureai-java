package com.secureai.model;

import lombok.Data;

import java.util.Map;

@Data
public class Topology {
    private String id;
    private Map<String, Node> nodes;
    private Map<String, Edge> edges;
}
