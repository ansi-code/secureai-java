package com.secureai.system;

import com.secureai.model.stateset.State;
import com.secureai.rl.abs.ArrayObservationSpace;
import lombok.Getter;

import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class SystemStateSpace extends ArrayObservationSpace<SystemState> {

    @Getter
    private final SystemEnvironment environment;

    @Getter
    private List<String> map;

    public SystemStateSpace(SystemEnvironment environment) {
        super(new int[]{environment.getSystemDefinition().getResources().size() * State.values().length});
        this.environment = environment;
        this.map = this.environment.getSystemDefinition().getResources().stream().flatMap(resourceId -> Stream.of(State.values()).map(stateId -> String.format("%s.%s", resourceId, stateId))).collect(Collectors.toList());
        System.out.println(this.map);

    }

    /*
    public SystemState encode(double[] input) {
        String systemActionId = map.getKey(a);
        String resourceId = systemActionId.substring(0, systemActionId.lastIndexOf('.'));
        String actionId = systemActionId.substring(systemActionId.lastIndexOf('.')+1);
        return new SystemState();
    }

    public Integer decode(SystemState systemState) {
        return map.get(String.format("%s.%s", systemAction.getResourceId(), systemAction.getActionId()));
    }
    */

}
