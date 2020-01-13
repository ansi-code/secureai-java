package com.secureai.model.topology;

import lombok.Data;

import java.util.Map;

@Data
public class Topology {
    private String id;
    private Map<String, Resource> resources;
    private Map<String, Connection> connections;
}
