package com.secureai.rl.rl4j;

import com.secureai.utils.RandomUtils;
import lombok.Getter;
import lombok.NonNull;
import lombok.RequiredArgsConstructor;

@RequiredArgsConstructor
public enum NodeAction {
    start(new NodeActionDefinition(200, 20, false,
            (state, i) -> !state.get(i, NodeState.active),
            (state, i) -> state.set(i, NodeState.active, RandomUtils.random.nextDouble() < 0.9))),

    restart(new NodeActionDefinition(300, 50, true,
            (state, i) -> state.get(i, NodeState.active) && state.get(i, NodeState.corrupted),
            (state, i) -> state.set(i, NodeState.corrupted, RandomUtils.random.nextDouble() < 0.4))),

    update(new NodeActionDefinition(1000, 150, false,
            (state, i) -> state.get(i, NodeState.active) && !state.get(i, NodeState.updated),
            (state, i) -> state.set(i, NodeState.updated, RandomUtils.random.nextDouble() < 0.9).set(i, NodeState.vulnerable, RandomUtils.random.nextDouble() < 0.2))),

    heal(new NodeActionDefinition(1200, 200, true,
            (state, i) -> state.get(i, NodeState.active) && state.get(i, NodeState.corrupted),
            (state, i) -> state.set(i, NodeState.corrupted, RandomUtils.random.nextDouble() < 0.1))),

    fix(new NodeActionDefinition(700, 100, true,
            (state, i) -> state.get(i, NodeState.active) && state.get(i, NodeState.vulnerable),
            (state, i) -> state.set(i, NodeState.vulnerable, RandomUtils.random.nextDouble() < 0.1))),

    heal2faster(NodeAction.heal.definition.multipliedBy(2)),
    heal2slower(NodeAction.heal.definition.multipliedBy(1d / 2));

    @Getter
    @NonNull
    private NodeActionDefinition definition;
}