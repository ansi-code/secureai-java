package com.secureai.system;

import com.secureai.model.stateset.State;
import com.secureai.rl.abs.TerminateFunction;

public class SystemTerminateFunction implements TerminateFunction<SystemState> {

    private SystemEnvironment environment;

    public SystemTerminateFunction(SystemEnvironment environment) {
        this.environment = environment;
    }

    @Override
    public boolean terminated(SystemState systemState) {
        for (String resourceId : this.environment.getSystemDefinition().getTopology().getResources().keySet())
            if (!systemState.get(resourceId, State.active) || !systemState.get(resourceId, State.updated) || systemState.get(resourceId, State.vulnerable) || systemState.get(resourceId, State.corrupted))
                return false;

        return true;
    }

}
