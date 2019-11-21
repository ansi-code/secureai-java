package com.secureai.system;

import lombok.AllArgsConstructor;
import lombok.Data;

@Data
@AllArgsConstructor
public class NodeActionDefinition {

    private double executionTime;
    private double executionCost;
    private boolean isDisruptive;
    private PreNodeStateFunction preNodeStateFunction;
    private PostNodeStateFunction postNodeStateFunction;

    public NodeActionDefinition multipliedBy(double factor) {
        return new NodeActionDefinition(this.executionTime * factor, this.executionCost * factor, this.isDisruptive, this.preNodeStateFunction, this.postNodeStateFunction);
    }

    public interface PreNodeStateFunction {
        boolean run(SystemState state, int i);
    }

    public interface PostNodeStateFunction {
        void run(SystemState state, int i);
    }

}