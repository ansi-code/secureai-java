package com.secureai.model.actionset;

import lombok.Data;

import java.util.Map;

@Data
public class ActionSet {
    private String id;

    private Map<String, Action> actions;
}
