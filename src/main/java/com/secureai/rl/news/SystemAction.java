package com.secureai.rl.news;

import lombok.AllArgsConstructor;
import lombok.Data;

@Data
@AllArgsConstructor
public class SystemAction {
    private Integer executionTime;
    private Integer executionCost;
    private Boolean isDisruptive;
    private PreSystemStateFunction preSystemStateFunction;
    private PostSystemStateFunction postSystemStateFunction;

    public interface PreSystemStateFunction {
        boolean run(SystemState state);
    }
    public interface PostSystemStateFunction {
        void run(SystemState state);
    }

    public void run(SystemState state) {
        if (this.preSystemStateFunction.run(state))
            this.postSystemStateFunction.run(state);
    }
}
