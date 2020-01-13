package com.secureai.model.actionset;

import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.databind.annotation.JsonDeserialize;
import com.secureai.system.SystemState;
import lombok.Data;

@Data
public class Action {
    @JsonProperty("execution-time")
    private Double executionTime;

    @JsonProperty("execution-cost")
    private Double executionCost;

    private Boolean disruptive;

    @JsonProperty("pre-condition")
    @JsonDeserialize(using = PreConditionDeserializer.class)
    private PreNodeStateFunction preCondition;

    @JsonProperty("post-condition")
    @JsonDeserialize(using = PostConditionDeserializer.class)
    private PostNodeStateFunction postCondition;

    public interface PreNodeStateFunction {
        boolean run(SystemState state, int i);
    }

    public interface PostNodeStateFunction {
        void run(SystemState state, int i);
    }
}
