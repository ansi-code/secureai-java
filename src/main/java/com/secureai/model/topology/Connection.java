package com.secureai.model.topology;

import lombok.Data;

@Data
public class Connection {

    private String type;
    private Direction direction;
    private String from;
    private String to;

    public enum Direction {
        unidirectional, bidirectional
    }
}

