package com.secureai.rl.news;

import com.secureai.model.Topology;
import lombok.Getter;
import org.deeplearning4j.rl4j.space.DiscreteSpace;

public class SystemActionSpace extends DiscreteSpace {

    public SystemActionSpace(Topology topology) {
        super(topology.getNodes().size() * NodeAction.values().length);
    }

    @Override
    public SystemAction encode(Integer a) {
        return NodeAction.values()[a].getAction();
    }

    public enum NodeAction {
        start(new SystemAction(200, 20, false, state -> true, state -> { })),
        restart(new SystemAction(300, 50, true, state -> true, state -> { })),
        update(new SystemAction(1000, 150, false, state -> true, state -> { })),
        heal(new SystemAction(1200, 200, true, state -> true, state -> { })),
        fix(new SystemAction(700, 100, true, state -> true, state -> { }));

        @Getter
        private SystemAction action;

        NodeAction(SystemAction action) {
            this.action = action;
        }
    }

}
