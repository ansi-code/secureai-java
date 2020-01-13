package com.secureai.system;

import com.secureai.model.actionset.Action;
import com.secureai.model.actionset.ActionSet;
import com.secureai.utils.IteratorUtils;
import lombok.Getter;
import org.deeplearning4j.rl4j.space.DiscreteSpace;

import java.util.stream.IntStream;
import java.util.stream.Stream;

public class SystemActionSpace extends DiscreteSpace {

    private final SystemDefinition systemDefinition;
    private final ActionSet actionSet;
    @Getter
    private double maxExecutionTime;
    @Getter
    private double maxExecutionCost;

    public SystemActionSpace(SystemDefinition systemDefinition, ActionSet actionSet) {
        super(systemDefinition.getTopology().getResources().size() * actionSet.getActions().size());
        this.systemDefinition = systemDefinition;
        this.actionSet = actionSet;
        this.maxExecutionTime = actionSet.getActions().values().stream().map(Action::getExecutionTime).max(Double::compareTo).orElse(0d);
        this.maxExecutionCost = actionSet.getActions().values().stream().map(Action::getExecutionCost).max(Double::compareTo).orElse(0d);
    }

    @Override
    public SystemAction encode(Integer a) {
        return new SystemAction(a / actionSet.getActions().size(), IteratorUtils.getInCollection(actionSet.getActions().values(), a % actionSet.getActions().size()));
    }

    public int size() {
        return this.size;
    }

    public Boolean[] actionsFilter(SystemState systemState) {
        return IntStream.range(0, systemDefinition.getTopology().getResources().size()).mapToObj(i -> this.actionsFilter(systemState, i)).flatMap(s -> s).toArray(Boolean[]::new);
    }

    public Stream<Boolean> actionsFilter(SystemState systemState, int i) {
        return this.actionSet.getActions().values().stream().map(a -> a.getPreCondition().run(systemState, i));
    }

}
