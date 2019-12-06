package com.secureai.system;

import com.secureai.model.Topology;
import lombok.Getter;
import org.deeplearning4j.rl4j.space.DiscreteSpace;

import java.util.Arrays;
import java.util.Comparator;
import java.util.stream.IntStream;
import java.util.stream.Stream;

public class SystemActionSpace extends DiscreteSpace {

    private final Topology topology;
    @Getter
    private double maxExecutionTime;
    @Getter
    private double maxExecutionCost;

    public SystemActionSpace(Topology topology) {
        super(topology.getNodes().size() * NodeAction.values().length);
        this.topology = topology;
        this.maxExecutionTime = Arrays.stream(NodeAction.values()).max(Comparator.comparingDouble(o -> o.getDefinition().getExecutionTime())).orElse(NodeAction.heal).getDefinition().getExecutionTime();
        this.maxExecutionCost = Arrays.stream(NodeAction.values()).max(Comparator.comparingDouble(o -> o.getDefinition().getExecutionCost())).orElse(NodeAction.heal).getDefinition().getExecutionCost();
    }

    @Override
    public SystemAction encode(Integer a) {
        return new SystemAction(a / NodeAction.values().length, NodeAction.values()[a % NodeAction.values().length]);
    }

    public int size() {
        return this.size;
    }

    public Boolean[] actionsFilter(SystemState systemState) {
        return IntStream.range(0, topology.getNodes().size()).mapToObj(i -> this.actionsFilter(systemState, i)).flatMap(s -> s).toArray(Boolean[]::new);
    }

    public Stream<Boolean> actionsFilter(SystemState systemState, int i) {
        return Arrays.stream(NodeAction.values()).map(a -> a.getDefinition().getPreNodeStateFunction().run(systemState, i));
    }

}
