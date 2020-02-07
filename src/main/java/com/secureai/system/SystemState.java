package com.secureai.system;

import com.secureai.model.stateset.State;
import com.secureai.rl.abs.DiscreteState;
import com.secureai.utils.RandomUtils;
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
        //this.random();
    }

    public void random() {
        this.environment.getSystemDefinition().getResources().forEach(resourceId -> {
            this.set(resourceId, State.active, RandomUtils.getRandom().nextDouble() < 0.7);
            this.set(resourceId, State.updated, RandomUtils.getRandom().nextDouble() < 0.5);
            this.set(resourceId, State.upgradable, RandomUtils.getRandom().nextDouble() > 0.5);
            this.set(resourceId, State.corrupted, RandomUtils.getRandom().nextDouble() > 0.6);
            this.set(resourceId, State.vulnerable, RandomUtils.getRandom().nextDouble() > 0.7);
        });
    }

    public void worst() {
        this.environment.getSystemDefinition().getResources().forEach(resourceId -> {
            this.set(resourceId, State.active, false);
            this.set(resourceId, State.updated, false);
            this.set(resourceId, State.upgradable, true);
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
