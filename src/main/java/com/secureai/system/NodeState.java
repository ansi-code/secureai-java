package com.secureai.system;

import lombok.Getter;
import lombok.NonNull;
import lombok.RequiredArgsConstructor;

@RequiredArgsConstructor
public enum NodeState {

    active(0),
    updated(1),
    corrupted(2),
    vulnerable(3);

    @Getter
    @NonNull
    private int value;

}
