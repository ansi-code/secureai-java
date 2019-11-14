package com.secureai.rl;

import lombok.AllArgsConstructor;
import lombok.Data;

@Data
@AllArgsConstructor
public class SystemAction {
    private Integer executionTime;
    private Integer executionCost;
    private Boolean isDisruptive;

    public interface PreSystemStateFunction {
        boolean run(SystemState state);
    }

    public interface PostSystemStateFunction {
        void run(SystemState state);
    }

    private PreSystemStateFunction preSystemStateFunction;
    private PostSystemStateFunction postSystemStateFunction;
}
