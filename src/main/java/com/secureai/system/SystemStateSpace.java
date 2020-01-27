package com.secureai.system;

import com.secureai.model.stateset.State;
import com.secureai.rl.abs.ArrayObservationSpace;
import lombok.Getter;

import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
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
    }

    public SystemState encode(Double[] input) {
        SystemState systemState = new SystemState(this.environment);
        IntStream.range(0, input.length).forEach(i -> {
            String systemStateId = this.map.get(i);
            Double systemStateValue = input[i];
            String resourceId = systemStateId.substring(0, systemStateId.lastIndexOf('.'));
            String stateId = systemStateId.substring(systemStateId.lastIndexOf('.') + 1);
            systemState.set(resourceId, State.valueOf(stateId), systemStateValue == 1);
        });
        return systemState;
    }

    public Double[] decode(SystemState systemState) {
        return this.map.stream().map(systemStateId -> {
            String resourceId = systemStateId.substring(0, systemStateId.lastIndexOf('.'));
            String stateId = systemStateId.substring(systemStateId.lastIndexOf('.') + 1);
            return systemState.get(resourceId, State.valueOf(stateId)) ? 1d : 0d;
        }).toArray(Double[]::new);
    }

}
