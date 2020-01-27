package com.secureai.system;

import lombok.Getter;
import org.deeplearning4j.rl4j.space.DiscreteSpace;

import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class SystemActionSpace extends DiscreteSpace {

    @Getter
    private final SystemEnvironment environment;

    @Getter
    private List<String> map;

    public SystemActionSpace(SystemEnvironment environment) {
        super(environment.getSystemDefinition().getResources().size() * environment.getActionSet().getActions().size());
        this.environment = environment;
        this.map = this.environment.getSystemDefinition().getResources().stream().flatMap(resourceId -> environment.getActionSet().getActions().keySet().stream().map(actionId -> String.format("%s.%s", resourceId, actionId))).collect(Collectors.toList());
    }

    @Override
    public SystemAction encode(Integer a) {
        String systemActionId = map.get(a);
        String resourceId = systemActionId.substring(0, systemActionId.lastIndexOf('.'));
        String actionId = systemActionId.substring(systemActionId.lastIndexOf('.') + 1);
        return new SystemAction(resourceId, actionId);
    }

    public Integer decode(SystemAction systemAction) {
        return map.indexOf(String.format("%s.%s", systemAction.getResourceId(), systemAction.getActionId()));
    }

    public Boolean[] actionsFilter(SystemState systemState) {
        return this.environment.getSystemDefinition().getResources().stream().map(i -> this.actionsFilter(systemState, i)).flatMap(s -> s).toArray(Boolean[]::new);
    }

    public Stream<Boolean> actionsFilter(SystemState systemState, String resourceId) {
        return this.environment.getActionSet().getActions().values().stream().map(a -> a.getPreCondition().run(systemState, resourceId));
    }

}
