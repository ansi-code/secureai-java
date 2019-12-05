package com.secureai.parser;

import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.databind.annotation.JsonDeserialize;
import com.secureai.system.NodeActionDefinition;
import lombok.Data;

@Data
public class Action {
    @JsonProperty("execution-time")
    private Integer executionTime;

    @JsonProperty("execution-cost")
    private Integer executionCost;

    private Boolean disruptive;

    @JsonProperty("pre-condition")
    @JsonDeserialize(using = PreConditionDeserializer.class)
    private NodeActionDefinition.PreNodeStateFunction preCondition;

    @JsonProperty("post-condition")
    @JsonDeserialize(using = PostConditionDeserializer.class)
    private NodeActionDefinition.PostNodeStateFunction postCondition;
}
