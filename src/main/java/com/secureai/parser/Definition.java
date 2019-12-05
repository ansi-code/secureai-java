package com.secureai.parser;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Data;

import java.util.Map;

@Data
public class Definition {
    private String id;

    @JsonProperty("state-variables")
    private Map<String, StateVariable> stateVariables;

    private Map<String, Action> actions;
}
