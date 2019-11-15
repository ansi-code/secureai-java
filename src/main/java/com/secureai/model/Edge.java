package com.secureai.model;

import lombok.Data;

@Data
public class Edge {

    private String type;
    private Direction direction;
    private String from;
    private String to;
    public enum Direction {
        unidirectional, bidirectional
    }
}

