package com.secureai.system;

import com.secureai.utils.IteratorUtils;
import lombok.Getter;
import org.deeplearning4j.rl4j.space.DiscreteSpace;

import java.util.stream.Stream;

public class SystemActionSpace extends DiscreteSpace {

    @Getter
    private final SystemEnvironment environment;

    public SystemActionSpace(SystemEnvironment environment) {
        super(environment.getSystemDefinition().getTopology().getResources().size() * environment.getActionSet().getActions().size());
        this.environment = environment;
    }

    @Override
    public SystemAction encode(Integer a) {
        return new SystemAction(
                this.environment.getSystemDefinition().getResourcesMap().getKey(a / environment.getActionSet().getActions().size()),
                IteratorUtils.getAtIndex(environment.getActionSet().getActions().values().iterator(), a % environment.getActionSet().getActions().size()));
    }

    public int size() {
        return this.size;
    }

    public Boolean[] actionsFilter(SystemState systemState) {
        return this.environment.getSystemDefinition().getTopology().getResources().keySet().stream().map(i -> this.actionsFilter(systemState, i)).flatMap(s -> s).toArray(Boolean[]::new);
    }

    public Stream<Boolean> actionsFilter(SystemState systemState, String resourceId) {
        return this.environment.getActionSet().getActions().values().stream().map(a -> a.getPreCondition().run(systemState, resourceId));
    }

}
