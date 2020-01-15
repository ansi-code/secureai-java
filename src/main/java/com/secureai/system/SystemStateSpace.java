package com.secureai.system;

import com.secureai.model.stateset.State;
import com.secureai.rl.abs.ArrayObservationSpace;

public class SystemStateSpace extends ArrayObservationSpace<SystemState> {

    public SystemStateSpace(SystemEnvironment environment) {
        super(new int[]{environment.getSystemDefinition().getResources().size() * State.values().length});
    }

    public int size() {
        return this.getShape()[0];
    }

}
