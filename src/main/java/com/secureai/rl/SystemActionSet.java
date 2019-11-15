package com.secureai.rl;

import lombok.Getter;

public enum SystemActionSet {
    start(new SystemAction(200, 20, false, state -> true, state -> {
    })),
    restart(new SystemAction(300, 50, true, state -> true, state -> {
    })),
    update(new SystemAction(1000, 150, false, state -> true, state -> {
    })),
    heal(new SystemAction(1200, 200, true, state -> true, state -> {
    })),
    fix(new SystemAction(700, 100, true, state -> true, state -> {
    }));

    @Getter
    private SystemAction action;

    SystemActionSet(SystemAction action) {
        this.action = action;
    }

}
