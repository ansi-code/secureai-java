package com.secureai.system;

import com.secureai.model.stateset.State;
import com.secureai.rl.abs.DiscreteState;
import org.apache.commons.lang3.ArrayUtils;

public class SystemState extends DiscreteState {

    private SystemEnvironment environment;

    public SystemState(SystemEnvironment environment) {
        super(environment.getSystemDefinition().getResources().size(), State.values().length);
        this.environment = environment;
    }

    @Override
    public void reset() {
        super.reset();

        this.worst();
    }

    public void random() {
        this.environment.getSystemDefinition().getResources().forEach(resourceId -> {
            this.set(resourceId, State.active, false);
            this.set(resourceId, State.updated, false);
            this.set(resourceId, State.corrupted, true);
            this.set(resourceId, State.vulnerable, true);
        });
    }

    public void worst() {
        this.environment.getSystemDefinition().getResources().forEach(resourceId -> {
            this.set(resourceId, State.active, false);
            this.set(resourceId, State.updated, false);
            this.set(resourceId, State.corrupted, true);
            this.set(resourceId, State.vulnerable, true);
        });
    }

    public boolean get(String resourceId, State state) {
        return this.get(this.environment.getSystemDefinition().getResources().indexOf(resourceId), state.getValue()) == 1;
    }

    public SystemState set(String resourceId, State state, boolean value) {
        this.set(value ? 1 : 0, this.environment.getSystemDefinition().getResources().indexOf(resourceId), state.getValue());
        return this;
    }

    public SystemState newInstance() {
        return new SystemState(this.environment);
    }

    @Override
    public double[] toArray() {
        return ArrayUtils.toPrimitive(this.environment.getObservationSpace().decode(this));
    }

}
