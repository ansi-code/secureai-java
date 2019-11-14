package com.secureai.model;

import lombok.Data;

@Data
public class Edge {

    public enum Direction {
        unidirectional, bidirectional
    }

    private String type;
    private Direction direction;
    private String from;
    private String to;
}

